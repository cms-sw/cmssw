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

// system include files
#include <array>
#include <vector>
#include <type_traits>
#include <atomic>
// user include files
#include "FWCore/Framework/interface/produce_helpers.h"
#include "FWCore/Framework/interface/EventSetupImpl.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "FWCore/Utilities/interface/ESIndices.h"
#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Concurrency/interface/WaitingTaskList.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ESParentContext.h"
#include "FWCore/ServiceRegistry/interface/ESModuleCallingContext.h"

namespace edm {
  void exceptionContext(cms::Exception&, ESModuleCallingContext const&);

  namespace eventsetup {
    class EventSetupRecordImpl;

    // The default decorator that does nothing
    template <typename TRecord>
    struct CallbackSimpleDecorator {
      void pre(const TRecord&) {}
      void post(const TRecord&) {}
    };

    template <typename T,          //producer's type
              typename TReturn,    //return type of the producer's method
              typename TRecord,    //the record passed in as an argument
              typename TDecorator  //allows customization using pre/post calls
              = CallbackSimpleDecorator<TRecord>>
    class Callback {
    public:
      using method_type = TReturn (T ::*)(const TRecord&);

      Callback(T* iProd, method_type iMethod, unsigned int iID, const TDecorator& iDec = TDecorator())
          : proxyData_{},
            producer_(iProd),
            callingContext_(&iProd->description()),
            method_(iMethod),
            id_(iID),
            wasCalledForThisRecord_(false),
            decorator_(iDec) {}

      Callback* clone() { return new Callback(producer_.get(), method_, id_, decorator_); }

      Callback(const Callback&) = delete;
      const Callback& operator=(const Callback&) = delete;

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
          callingContext_.setContext(ESModuleCallingContext::State::kPrefetching, iParent);
          iRecord->activityRegistry()->preESModulePrefetchingSignal_.emit(iRecord->key(), callingContext_);
          if UNLIKELY (producer_->hasMayConsumes()) {
            //after prefetching need to do the mayGet
            ServiceWeakToken weakToken = token;
            auto mayGetTask = edm::make_waiting_task(
                [this, iRecord, iEventSetupImpl, weakToken, group](std::exception_ptr const* iExcept) {
                  if (iExcept) {
                    runProducerAsync(group, iExcept, iRecord, iEventSetupImpl, weakToken.lock());
                    return;
                  }
                  if (handleMayGet(iRecord, iEventSetupImpl)) {
                    auto runTask = edm::make_waiting_task(
                        [this, group, iRecord, iEventSetupImpl, weakToken](std::exception_ptr const* iExcept) {
                          runProducerAsync(group, iExcept, iRecord, iEventSetupImpl, weakToken.lock());
                        });
                    prefetchNeededDataAsync(WaitingTaskHolder(*group, runTask),
                                            iEventSetupImpl,
                                            &((*postMayGetProxies_).front()),
                                            weakToken.lock());
                  } else {
                    runProducerAsync(group, iExcept, iRecord, iEventSetupImpl, weakToken.lock());
                  }
                });

            //Get everything we can before knowing about the mayGets
            prefetchNeededDataAsync(WaitingTaskHolder(*group, mayGetTask), iEventSetupImpl, getTokenIndices(), token);
          } else {
            ServiceWeakToken weakToken = token;
            auto task = edm::make_waiting_task(
                [this, group, iRecord, iEventSetupImpl, weakToken](std::exception_ptr const* iExcept) {
                  runProducerAsync(group, iExcept, iRecord, iEventSetupImpl, weakToken.lock());
                });
            prefetchNeededDataAsync(WaitingTaskHolder(*group, task), iEventSetupImpl, getTokenIndices(), token);
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

      void runProducerAsync(oneapi::tbb::task_group* iGroup,
                            std::exception_ptr const* iExcept,
                            EventSetupRecordImpl const* iRecord,
                            EventSetupImpl const* iEventSetupImpl,
                            ServiceToken const& token) {
        if (iExcept) {
          //The cache held by the CallbackProxy was already set to invalid at the beginning of the IOV
          taskList_.doneWaiting(*iExcept);
          return;
        }
        iRecord->activityRegistry()->postESModulePrefetchingSignal_.emit(iRecord->key(), callingContext_);
        ServiceWeakToken weakToken = token;
        producer_->queue().push(*iGroup, [this, iRecord, iEventSetupImpl, weakToken]() {
          callingContext_.setState(ESModuleCallingContext::State::kRunning);
          std::exception_ptr exceptPtr;
          try {
            convertException::wrap([this, iRecord, iEventSetupImpl, weakToken] {
              auto proxies = getTokenIndices();
              if (postMayGetProxies_) {
                proxies = &((*postMayGetProxies_).front());
              }
              TRecord rec;
              edm::ESParentContext pc{&callingContext_};
              rec.setImpl(iRecord, transitionID(), proxies, iEventSetupImpl, &pc);
              ServiceRegistry::Operate operate(weakToken.lock());
              iRecord->activityRegistry()->preESModuleSignal_.emit(iRecord->key(), callingContext_);
              struct EndGuard {
                EndGuard(EventSetupRecordImpl const* iRecord, ESModuleCallingContext const& iContext)
                    : record_{iRecord}, context_{iContext} {}
                ~EndGuard() { record_->activityRegistry()->postESModuleSignal_.emit(record_->key(), context_); }
                EventSetupRecordImpl const* record_;
                ESModuleCallingContext const& context_;
              };
              EndGuard guard(iRecord, callingContext_);
              decorator_.pre(rec);
              storeReturnedValues((producer_->*method_)(rec));
              decorator_.post(rec);
            });
          } catch (cms::Exception& iException) {
            edm::exceptionContext(iException, callingContext_);
            exceptPtr = std::current_exception();
          }
          taskList_.doneWaiting(exceptPtr);
        });
      }

      std::array<void*, produce::size<TReturn>::value> proxyData_;
      std::optional<std::vector<ESProxyIndex>> postMayGetProxies_;
      edm::propagate_const<T*> producer_;
      ESModuleCallingContext callingContext_;
      edm::WaitingTaskList taskList_;
      method_type method_;
      // This transition id identifies which setWhatProduced call this Callback is associated with
      const unsigned int id_;
      std::atomic<bool> wasCalledForThisRecord_;
      TDecorator decorator_;
    };
  }  // namespace eventsetup
}  // namespace edm

#endif
