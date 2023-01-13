// -*- C++ -*-
#ifndef FWCore_Framework_CallbackBase_h
#define FWCore_Framework_CallbackBase_h
//
// Package:     Framework
// Class  :     CallbackBase
//
/**\class edm::eventsetup::CallbackBase

 Description: Functional object used as the 'callback' for the CallbackProxy

 Usage: Produces data objects for ESProducers in EventSetup system

*/
//
// Author:      Chris Jones (original author, this was part of Callback.h),
//              W. David Dagenhart (Refactored version + CallbackExternalWork, 2023)
// Created:     Sun Apr 17 14:30:24 EDT 2005
//

#include <array>
#include <atomic>
#include <exception>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "oneapi/tbb/task_group.h"

#include "FWCore/Concurrency/interface/SerialTaskQueueChain.h"
#include "FWCore/Concurrency/interface/WaitingTask.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Concurrency/interface/WaitingTaskList.h"
#include "FWCore/Framework/interface/EventSetupImpl.h"
#include "FWCore/Framework/interface/EventSetupRecordImpl.h"
#include "FWCore/Framework/interface/produce_helpers.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ESModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ESParentContext.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Utilities/interface/ESIndices.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/Likely.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "FWCore/Utilities/interface/Signal.h"

namespace edm {
  void exceptionContext(cms::Exception&, ESModuleCallingContext const&);

  namespace eventsetup {

    // The default decorator that does nothing
    template <typename TRecord>
    struct CallbackSimpleDecorator {
      void pre(const TRecord&) {}
      void post(const TRecord&) {}
    };

    template <typename T,             //producer's type
              typename TProduceFunc,  //produce functor type
              typename TReturn,       //return type of the producer's method
              typename TRecord,       //the record passed in as an argument
              typename TDecorator>    //allows customization using pre/post calls
    class CallbackBase {
    public:
      CallbackBase(T* iProd, std::shared_ptr<TProduceFunc> iProduceFunc, unsigned int iID, const TDecorator& iDec)
          : proxyData_{},
            producer_(iProd),
            callingContext_(&iProd->description()),
            produceFunction_(std::move(iProduceFunc)),
            id_(iID),
            wasCalledForThisRecord_(false),
            decorator_(iDec) {}

      CallbackBase(const CallbackBase&) = delete;
      CallbackBase& operator=(const CallbackBase&) = delete;
      CallbackBase(CallbackBase&&) = delete;
      CallbackBase& operator=(CallbackBase&&) = delete;

      WaitingTaskHolder makeProduceTask(oneapi::tbb::task_group* group,
                                        ServiceWeakToken const& serviceToken,
                                        EventSetupRecordImpl const* record,
                                        EventSetupImpl const* eventSetupImpl,
                                        bool emitPostPrefetchingSignal) {
        return WaitingTaskHolder(
            *group,
            make_waiting_task([this, group, serviceToken, record, eventSetupImpl, emitPostPrefetchingSignal](
                                  std::exception_ptr const* iException) {
              std::exception_ptr excptr;
              if (iException) {
                excptr = *iException;
              }
              if (emitPostPrefetchingSignal) {
                try {
                  convertException::wrap([this, &serviceToken, &record] {
                    ServiceRegistry::Operate guard(serviceToken.lock());
                    record->activityRegistry()->postESModulePrefetchingSignal_.emit(record->key(), callingContext_);
                  });
                } catch (cms::Exception& caughtException) {
                  if (not excptr) {
                    exceptionContext(caughtException, callingContext_);
                    excptr = std::current_exception();
                  }
                }
              }
              if (excptr) {
                taskList_.doneWaiting(excptr);
                return;
              }

              producer_->queue().push(*group, [this, serviceToken, record, eventSetupImpl]() {
                callingContext_.setState(ESModuleCallingContext::State::kRunning);
                std::exception_ptr exceptPtr;
                try {
                  convertException::wrap([this, &serviceToken, &record, &eventSetupImpl] {
                    ESModuleCallingContext const& context = callingContext_;
                    auto proxies = getTokenIndices();
                    if (postMayGetProxies_) {
                      proxies = &((*postMayGetProxies_).front());
                    }
                    TRecord rec;
                    ESParentContext pc{&context};
                    rec.setImpl(record, transitionID(), proxies, eventSetupImpl, &pc);
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
                    decorator_.pre(rec);
                    storeReturnedValues((*produceFunction_)(rec));
                    decorator_.post(rec);
                  });
                } catch (cms::Exception& iException) {
                  exceptionContext(iException, callingContext_);
                  exceptPtr = std::current_exception();
                }
                taskList_.doneWaiting(exceptPtr);
              });
            }));
      }

      template <typename RunModuleFnctr>
      void prefetchAsyncImpl(RunModuleFnctr&& runModuleFnctr,
                             WaitingTaskHolder iTask,
                             EventSetupRecordImpl const* iRecord,
                             EventSetupImpl const* iEventSetupImpl,
                             ServiceToken const& token,
                             ESParentContext const& iParent) {
        bool expected = false;
        auto doPrefetch = wasCalledForThisRecord_.compare_exchange_strong(expected, true);
        taskList_.add(iTask);
        if (doPrefetch) {
          auto group = iTask.group();
          ServiceWeakToken weakToken(token);
          WaitingTaskHolder runModuleTaskHolder = runModuleFnctr(group, weakToken, iRecord, iEventSetupImpl);
          callingContext_.setContext(ESModuleCallingContext::State::kPrefetching, iParent);
          iRecord->activityRegistry()->preESModulePrefetchingSignal_.emit(iRecord->key(), callingContext_);
          if UNLIKELY (producer_->hasMayConsumes()) {
            //after prefetching need to do the mayGet
            auto mayGetTask = make_waiting_task(
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

      std::optional<std::vector<ESProxyIndex>> const& postMayGetProxies() const { return postMayGetProxies_; }
      T* producer() { return producer_.get(); }
      ESModuleCallingContext& callingContext() { return callingContext_; }
      WaitingTaskList& taskList() { return taskList_; }
      std::shared_ptr<TProduceFunc> const& produceFunction() { return produceFunction_; }
      TDecorator const& decorator() const { return decorator_; }
      SerialTaskQueueChain& queue() { return producer_->queue(); }

    protected:
      ~CallbackBase() = default;

    private:
      void storeReturnedValues(TReturn iReturn) {
        using type = typename produce::product_traits<TReturn>::type;
        setData<typename type::head_type, typename type::tail_type>(iReturn);
      }

      void prefetchNeededDataAsync(WaitingTaskHolder task,
                                   EventSetupImpl const* iImpl,
                                   ESProxyIndex const* proxies,
                                   ServiceToken const& token) const {
        auto recs = producer_->getTokenRecordIndices(id_);
        auto n = producer_->numberOfTokenIndices(id_);
        for (size_t i = 0; i != n; ++i) {
          auto rec = iImpl->findImpl(recs[i]);
          if (rec) {
            rec->prefetchAsync(task, proxies[i], iImpl, token, ESParentContext{&callingContext_});
          }
        }
      }

      bool handleMayGet(EventSetupRecordImpl const* iRecord, EventSetupImpl const* iEventSetupImpl) {
        //Handle mayGets
        TRecord rec;
        ESParentContext pc{&callingContext_};
        rec.setImpl(iRecord, transitionID(), getTokenIndices(), iEventSetupImpl, &pc);
        postMayGetProxies_ = producer_->updateFromMayConsumes(id_, rec);
        return static_cast<bool>(postMayGetProxies_);
      }

      std::array<void*, produce::size<TReturn>::value> proxyData_;
      std::optional<std::vector<ESProxyIndex>> postMayGetProxies_;
      propagate_const<T*> producer_;
      ESModuleCallingContext callingContext_;
      WaitingTaskList taskList_;
      // Using std::shared_ptr in order to share the state of the
      // functors across all clones
      std::shared_ptr<TProduceFunc> produceFunction_;
      // This transition id identifies which setWhatProduced call this Callback is associated with
      const unsigned int id_;
      std::atomic<bool> wasCalledForThisRecord_;
      TDecorator decorator_;
    };
  }  // namespace eventsetup
}  // namespace edm

#endif
