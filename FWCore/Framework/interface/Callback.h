#ifndef FWCore_Framework_Callback_h
#define FWCore_Framework_Callback_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     Callback
//
/**\class edm::eventsetup::Callback

 Description: Functional object used as the 'callback' for the CallbackProxy

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Sun Apr 17 14:30:24 EDT 2005
//

#include <array>
#include <exception>
#include <memory>
#include <optional>
#include <utility>
#include <vector>
#include <type_traits>
#include <atomic>

#include "oneapi/tbb/task_group.h"

#include "FWCore/Framework/interface/produce_helpers.h"
#include "FWCore/Framework/interface/EventSetupImpl.h"
#include "FWCore/Framework/interface/EventSetupRecordImpl.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "FWCore/Utilities/interface/ESIndices.h"
#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Utilities/interface/Signal.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "FWCore/Concurrency/interface/SerialTaskQueueChain.h"
#include "FWCore/Concurrency/interface/WaitingTask.h"
#include "FWCore/Concurrency/interface/WaitingTaskList.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ESParentContext.h"
#include "FWCore/ServiceRegistry/interface/ESModuleCallingContext.h"

namespace edm {
  void exceptionContext(cms::Exception&, ESModuleCallingContext const&);

  namespace eventsetup {

    // The default decorator that does nothing
    template <typename TRecord>
    struct CallbackSimpleDecorator {
      void pre(const TRecord&) {}
      void post(const TRecord&) {}
    };

    template <typename TRecord>
    class DummyAcquireFunctor {
    public:
      void operator()(TRecord const&, WaitingTaskWithArenaHolder) {}
    };

    template <typename T,             //producer's type
              typename TAcquireFunc,  //acquire functor type
              typename TProduceFunc,  //produce functor type
              typename TReturn,       //return type of the producer's method
              typename TRecord,       //the record passed in as an argument
              typename TDecorator     //allows customization using pre/post calls
              = CallbackSimpleDecorator<TRecord>>
    class Callback {
    public:
      Callback(T* iProd, TProduceFunc iProduceFunc, unsigned int iID, const TDecorator& iDec = TDecorator())
          : Callback(iProd,
                     std::shared_ptr<TAcquireFunc>(),
                     std::make_shared<TProduceFunc>(std::move(iProduceFunc)),
                     iID,
                     iDec) {}

      Callback(T* iProd,
               TAcquireFunc iAcquireFunc,
               TProduceFunc iProduceFunc,
               unsigned int iID,
               const TDecorator& iDec = TDecorator())
          : Callback(iProd,
                     std::make_shared<TAcquireFunc>(std::move(iAcquireFunc)),
                     std::make_shared<TProduceFunc>(std::move(iProduceFunc)),
                     iID,
                     iDec) {}

      Callback(T* iProd,
               std::shared_ptr<TAcquireFunc> iAcquireFunc,
               std::shared_ptr<TProduceFunc> iProduceFunc,
               unsigned int iID,
               const TDecorator& iDec = TDecorator())
          : proxyData_{},
            producer_(iProd),
            callingContext_(&iProd->description()),
            acquireFunction_(std::move(iAcquireFunc)),
            produceFunction_(std::move(iProduceFunc)),
            id_(iID),
            wasCalledForThisRecord_(false),
            decorator_(iDec) {}

      Callback* clone() { return new Callback(producer_.get(), acquireFunction_, produceFunction_, id_, decorator_); }

      Callback(const Callback&) = delete;
      const Callback& operator=(const Callback&) = delete;

      class ESProduceTask : public WaitingTask {
      public:
        ESProduceTask(Callback* iCallback,
                      oneapi::tbb::task_group* iGroup,
                      ServiceToken const& iServiceToken,
                      EventSetupRecordImpl const* iRecord,
                      EventSetupImpl const* iEventSetupImpl)
            : callback_(iCallback),
              group_(iGroup),
              serviceToken_(iServiceToken),
              record_(iRecord),
              eventSetupImpl_(iEventSetupImpl) {}

        void execute() final {
          auto excptr = exceptionPtr();
          if (!callback_->acquireFunction_) {
            CMS_SA_ALLOW try {
              convertException::wrap([this] {
                ServiceRegistry::Operate guard(serviceToken_.lock());
                record_->activityRegistry()->postESModulePrefetchingSignal_.emit(record_->key(),
                                                                                 callback_->callingContext_);
              });
            } catch (cms::Exception& iException) {
              if (not excptr) {
                edm::exceptionContext(iException, callback_->callingContext_);
                excptr = std::current_exception();
              }
            }
          }
          if (excptr) {
            callback_->taskList_.doneWaiting(excptr);
            return;
          }

          callback_->producer_->queue().push(
              *group_,
              [callback = callback_,
               serviceToken = std::move(serviceToken_),
               record = record_,
               eventSetupImpl = eventSetupImpl_]() {
                callback->callingContext_.setState(ESModuleCallingContext::State::kRunning);
                std::exception_ptr exceptPtr;
                try {
                  convertException::wrap([&callback, &serviceToken, &record, &eventSetupImpl] {
                    ESModuleCallingContext const& context = callback->callingContext_;
                    auto proxies = callback->getTokenIndices();
                    if (callback->postMayGetProxies_) {
                      proxies = &((*callback->postMayGetProxies_).front());
                    }
                    TRecord rec;
                    edm::ESParentContext pc{&context};
                    rec.setImpl(record, callback->transitionID(), proxies, eventSetupImpl, &pc);
                    ServiceRegistry::Operate operate(serviceToken.lock());
                    record->activityRegistry()->preESModuleSignal_.emit(record->key(), context);
                    struct EndGuard {
                      EndGuard(EventSetupRecordImpl const* iRecord, ESModuleCallingContext const& iContext)
                          : record_{iRecord}, context_{iContext} {}
                      ~EndGuard() { record_->activityRegistry()->postESModuleSignal_.emit(record_->key(), context_); }
                      EventSetupRecordImpl const* record_;
                      ESModuleCallingContext const& context_;
                    };
                    EndGuard guard(record, context);
                    callback->decorator_.pre(rec);
                    callback->storeReturnedValues((*callback->produceFunction_)(rec));
                    callback->decorator_.post(rec);
                  });
                } catch (cms::Exception& iException) {
                  edm::exceptionContext(iException, callback->callingContext_);
                  exceptPtr = std::current_exception();
                }
                callback->taskList_.doneWaiting(exceptPtr);
              });
        }

      private:
        Callback* callback_;
        oneapi::tbb::task_group* group_;
        ServiceWeakToken serviceToken_;
        EventSetupRecordImpl const* record_;
        EventSetupImpl const* eventSetupImpl_;
      };

      class ESAcquireTask : public WaitingTask {
      public:
        ESAcquireTask(Callback* iCallback,
                      WaitingTaskWithArenaHolder iHolder,
                      ServiceToken const& iServiceToken,
                      EventSetupRecordImpl const* iRecord,
                      EventSetupImpl const* iEventSetupImpl)
            : callback_(iCallback),
              holder_(std::move(iHolder)),
              serviceToken_(iServiceToken),
              record_(iRecord),
              eventSetupImpl_(iEventSetupImpl) {}

        void execute() final {
          auto excptr = exceptionPtr();
          CMS_SA_ALLOW try {
            convertException::wrap([this] {
              ServiceRegistry::Operate guard(serviceToken_.lock());
              record_->activityRegistry()->postESModulePrefetchingSignal_.emit(record_->key(),
                                                                               callback_->callingContext_);
            });
          } catch (cms::Exception& iException) {
            if (not excptr) {
              edm::exceptionContext(iException, callback_->callingContext_);
              excptr = std::current_exception();
            }
          }
          if (excptr) {
            holder_.doneWaiting(excptr);
            return;
          }

          auto group = holder_.group();
          callback_->producer_->queue().push(
              *group,
              [callback = callback_,
               holder = std::move(holder_),
               serviceToken = std::move(serviceToken_),
               record = record_,
               eventSetupImpl = eventSetupImpl_]() mutable {
                callback->callingContext_.setState(ESModuleCallingContext::State::kRunning);
                std::exception_ptr exceptPtr;
                try {
                  convertException::wrap([&callback, &holder, &serviceToken, &record, &eventSetupImpl] {
                    ESModuleCallingContext const& context = callback->callingContext_;
                    auto proxies = callback->getTokenIndices();
                    if (callback->postMayGetProxies_) {
                      proxies = &((*callback->postMayGetProxies_).front());
                    }
                    TRecord rec;
                    edm::ESParentContext pc{&context};
                    rec.setImpl(record, callback->transitionID(), proxies, eventSetupImpl, &pc);
                    ServiceRegistry::Operate operate(serviceToken.lock());
                    record->activityRegistry()->preESModuleAcquireSignal_.emit(record->key(), context);
                    struct EndGuard {
                      EndGuard(EventSetupRecordImpl const* iRecord, ESModuleCallingContext const& iContext)
                          : record_{iRecord}, context_{iContext} {}
                      ~EndGuard() {
                        record_->activityRegistry()->postESModuleAcquireSignal_.emit(record_->key(), context_);
                      }
                      EventSetupRecordImpl const* record_;
                      ESModuleCallingContext const& context_;
                    };
                    EndGuard guard(record, context);
                    (*callback->acquireFunction_)(rec, holder);
                  });
                } catch (cms::Exception& iException) {
                  iException.addContext("Running acquire");
                  exceptPtr = std::current_exception();
                }
                holder.doneWaiting(exceptPtr);
              });
        }

      private:
        Callback* callback_;
        WaitingTaskWithArenaHolder holder_;
        ServiceWeakToken serviceToken_;
        EventSetupRecordImpl const* record_;
        EventSetupImpl const* eventSetupImpl_;
      };

      class HandleESExternalWorkExceptionTask : public WaitingTask {
      public:
        HandleESExternalWorkExceptionTask(Callback* callback,
                                          oneapi::tbb::task_group* group,
                                          WaitingTask* esProduceTask)
            : callback_(callback), group_(group), esProduceTask_(esProduceTask) {}

        void execute() final {
          auto excptr = exceptionPtr();
          WaitingTaskHolder holder(*group_, esProduceTask_);
          if (excptr) {
            try {
              convertException::wrap([excptr]() { std::rethrow_exception(excptr); });
            } catch (cms::Exception& exception) {
              exception.addContext("Running acquire and external work");
              edm::exceptionContext(exception, callback_->callingContext_);
              holder.doneWaiting(std::current_exception());
            }
          }
        }

      private:
        Callback* callback_;
        oneapi::tbb::task_group* group_;
        WaitingTask* esProduceTask_;
      };

      void prefetchAsync(WaitingTaskHolder iTask,
                         EventSetupRecordImpl const* iRecord,
                         EventSetupImpl const* iEventSetupImpl,
                         ServiceToken const& token,
                         ESParentContext const& iParent) {
        bool expected = false;
        auto doPrefetch = wasCalledForThisRecord_.compare_exchange_strong(expected, true);
        taskList_.add(iTask);
        auto group = iTask.group();
        if (doPrefetch) {
          WaitingTask* runModuleTask = new ESProduceTask(this, group, token, iRecord, iEventSetupImpl);
          if (acquireFunction_) {
            WaitingTaskWithArenaHolder waitingTaskWithArenaHolder(
                *group, new HandleESExternalWorkExceptionTask(this, group, runModuleTask));
            runModuleTask =
                new ESAcquireTask(this, std::move(waitingTaskWithArenaHolder), token, iRecord, iEventSetupImpl);
          }
          WaitingTaskHolder runModuleTaskHolder(*group, runModuleTask);

          callingContext_.setContext(ESModuleCallingContext::State::kPrefetching, iParent);
          iRecord->activityRegistry()->preESModulePrefetchingSignal_.emit(iRecord->key(), callingContext_);
          if UNLIKELY (producer_->hasMayConsumes()) {
            //after prefetching need to do the mayGet
            ServiceWeakToken weakToken = token;
            auto mayGetTask = edm::make_waiting_task(
                [this, iRecord, iEventSetupImpl, weakToken, runModuleTaskHolder = std::move(runModuleTaskHolder)](
                    std::exception_ptr const* iExcept) mutable {
                  if (iExcept) {
                    runModuleTaskHolder.doneWaiting(*iExcept);
                    return;
                  }
                  if (handleMayGet(iRecord, iEventSetupImpl)) {
                    prefetchNeededDataAsync(
                        runModuleTaskHolder, iEventSetupImpl, &((*postMayGetProxies_).front()), weakToken.lock());
                  } else {
                    runModuleTaskHolder.doneWaiting(std::exception_ptr{});
                  }
                });

            //Get everything we can before knowing about the mayGets
            prefetchNeededDataAsync(WaitingTaskHolder(*group, mayGetTask), iEventSetupImpl, getTokenIndices(), token);
          } else {
            prefetchNeededDataAsync(runModuleTaskHolder, iEventSetupImpl, getTokenIndices(), token);
          }
        }
      }

      template <class DataT>
      void holdOntoPointer(DataT* iData) {
        proxyData_[produce::find_index<TReturn, DataT>::value] = iData;
      }

      void storeReturnedValues(TReturn iReturn) {
        using type = typename produce::product_traits<TReturn>::type;
        setData<typename type::head_type, typename type::tail_type>(iReturn);
      }

      template <class RemainingContainerT, class DataT, class ProductsT>
      void setData(ProductsT& iProducts) {
        DataT* temp = reinterpret_cast<DataT*>(proxyData_[produce::find_index<TReturn, DataT>::value]);
        if (nullptr != temp) {
          moveFromTo(iProducts, *temp);
        }
        if constexpr (not std::is_same_v<produce::Null, RemainingContainerT>) {
          setData<typename RemainingContainerT::head_type, typename RemainingContainerT::tail_type>(iProducts);
        }
      }
      void newRecordComing() {
        wasCalledForThisRecord_ = false;
        taskList_.reset();
      }

      unsigned int transitionID() const { return id_; }
      ESProxyIndex const* getTokenIndices() const { return producer_->getTokenIndices(id_); }

    private:
      void prefetchNeededDataAsync(WaitingTaskHolder task,
                                   EventSetupImpl const* iImpl,
                                   ESProxyIndex const* proxies,
                                   edm::ServiceToken const& token) const {
        auto recs = producer_->getTokenRecordIndices(id_);
        auto n = producer_->numberOfTokenIndices(id_);
        for (size_t i = 0; i != n; ++i) {
          auto rec = iImpl->findImpl(recs[i]);
          if (rec) {
            rec->prefetchAsync(task, proxies[i], iImpl, token, edm::ESParentContext{&callingContext_});
          }
        }
      }

      bool handleMayGet(EventSetupRecordImpl const* iRecord, EventSetupImpl const* iEventSetupImpl) {
        //Handle mayGets
        TRecord rec;
        edm::ESParentContext pc{&callingContext_};
        rec.setImpl(iRecord, transitionID(), getTokenIndices(), iEventSetupImpl, &pc);
        postMayGetProxies_ = producer_->updateFromMayConsumes(id_, rec);
        return static_cast<bool>(postMayGetProxies_);
      }

      std::array<void*, produce::size<TReturn>::value> proxyData_;
      std::optional<std::vector<ESProxyIndex>> postMayGetProxies_;
      edm::propagate_const<T*> producer_;
      ESModuleCallingContext callingContext_;
      edm::WaitingTaskList taskList_;
      // Using std::shared_ptr in order to share the state of the
      // functors across all clones
      std::shared_ptr<TAcquireFunc> acquireFunction_;
      std::shared_ptr<TProduceFunc> produceFunction_;
      // This transition id identifies which setWhatProduced call this Callback is associated with
      const unsigned int id_;
      std::atomic<bool> wasCalledForThisRecord_;
      TDecorator decorator_;
    };
  }  // namespace eventsetup
}  // namespace edm

#endif
