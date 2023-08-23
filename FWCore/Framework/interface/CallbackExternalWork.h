// -*- C++ -*-
#ifndef FWCore_Framework_CallbackExternalWork_h
#define FWCore_Framework_CallbackExternalWork_h
//
// Package:     Framework
// Class  :     CallbackExternalWork
//
/**\class edm::eventsetup::CallbackExternalWork

 Description: Functional object used as the 'callback' for the CallbackProductResolver

 Usage: Produces data objects for ESProducers in EventSetup system

*/
//
// Author:      W. David Dagenhart
// Created:     27 February 2023

#include <exception>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "oneapi/tbb/task_group.h"

#include "FWCore/Concurrency/interface/SerialTaskQueueChain.h"
#include "FWCore/Concurrency/interface/WaitingTask.h"
#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Concurrency/interface/WaitingTaskList.h"
#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/Framework/interface/CallbackBase.h"
#include "FWCore/Framework/interface/EventSetupRecordImpl.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ESModuleCallingContext.h"
#include "FWCore/ServiceRegistry/interface/ESParentContext.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/Signal.h"

namespace edm {

  class EventSetupImpl;

  namespace eventsetup {

    namespace impl {
      template <typename U>
      struct AcquireCacheType {
        using type = std::optional<U>;
        static U& value(type& val) { return val.value(); }
      };
      template <typename U>
      struct AcquireCacheType<std::optional<U>> {
        using type = std::optional<U>;
        static std::optional<U>& value(type& val) { return val; }
      };
      template <typename U>
      struct AcquireCacheType<std::unique_ptr<U>> {
        using type = std::unique_ptr<U>;
        static std::unique_ptr<U>& value(type& val) { return val; }
      };
      template <typename U>
      struct AcquireCacheType<std::shared_ptr<U>> {
        using type = std::shared_ptr<U>;
        static std::shared_ptr<U>& value(type& val) { return val; }
      };
    }  // namespace impl

    template <typename T,               //producer's type
              typename TAcquireFunc,    //acquire functor type
              typename TAcquireReturn,  //return type of the acquire method
              typename TProduceFunc,    //produce functor type
              typename TProduceReturn,  //return type of the produce method
              typename TRecord,         //the record passed in as an argument
              typename TDecorator       //allows customization using pre/post calls
              = CallbackSimpleDecorator<TRecord>>
    class CallbackExternalWork : public CallbackBase<T, TProduceFunc, TProduceReturn, TRecord, TDecorator> {
    public:
      using Base = CallbackBase<T, TProduceFunc, TProduceReturn, TRecord, TDecorator>;

      CallbackExternalWork(T* iProd,
                           TAcquireFunc iAcquireFunc,
                           TProduceFunc iProduceFunc,
                           unsigned int iID,
                           const TDecorator& iDec = TDecorator())
          : CallbackExternalWork(iProd,
                                 std::make_shared<TAcquireFunc>(std::move(iAcquireFunc)),
                                 std::make_shared<TProduceFunc>(std::move(iProduceFunc)),
                                 iID,
                                 iDec) {}

      CallbackExternalWork* clone() {
        return new CallbackExternalWork(
            Base::producer(), acquireFunction_, Base::produceFunction(), Base::transitionID(), Base::decorator());
      }

      void prefetchAsync(WaitingTaskHolder iTask,
                         EventSetupRecordImpl const* iRecord,
                         EventSetupImpl const* iEventSetupImpl,
                         ServiceToken const& token,
                         ESParentContext const& iParent) {
        return Base::prefetchAsyncImpl(
            [this](auto&& group, auto&& token, auto&& record, auto&& es) {
              constexpr bool emitPostPrefetchingSignal = false;
              auto produceFunctor = [this](TRecord const& record) {
                auto returnValue = (*Base::produceFunction())(
                    record, std::move(impl::AcquireCacheType<TAcquireReturn>::value(acquireCache_)));
                acquireCache_.reset();
                return returnValue;
              };
              WaitingTaskHolder produceTask =
                  Base::makeProduceTask(group, token, record, es, emitPostPrefetchingSignal, std::move(produceFunctor));

              WaitingTaskWithArenaHolder waitingTaskWithArenaHolder =
                  makeExceptionHandlerTask(std::move(produceTask), group);

              return makeAcquireTask(std::move(waitingTaskWithArenaHolder), group, token, record, es);
            },
            std::move(iTask),
            iRecord,
            iEventSetupImpl,
            token,
            iParent);
      }

    private:
      CallbackExternalWork(T* iProd,
                           std::shared_ptr<TAcquireFunc> iAcquireFunc,
                           std::shared_ptr<TProduceFunc> iProduceFunc,
                           unsigned int iID,
                           const TDecorator& iDec = TDecorator())
          : Base(iProd, std::move(iProduceFunc), iID, iDec), acquireFunction_(std::move(iAcquireFunc)) {}

      WaitingTaskHolder makeAcquireTask(WaitingTaskWithArenaHolder waitingTaskWithArenaHolder,
                                        oneapi::tbb::task_group* group,
                                        ServiceWeakToken const& serviceToken,
                                        EventSetupRecordImpl const* record,
                                        EventSetupImpl const* eventSetupImpl) {
        return WaitingTaskHolder(
            *group,
            make_waiting_task(
                [this, holder = std::move(waitingTaskWithArenaHolder), group, serviceToken, record, eventSetupImpl](
                    std::exception_ptr const* iException) mutable {
                  std::exception_ptr excptr;
                  if (iException) {
                    excptr = *iException;
                  }
                  try {
                    convertException::wrap([this, &serviceToken, &record] {
                      ServiceRegistry::Operate guard(serviceToken.lock());
                      record->activityRegistry()->postESModulePrefetchingSignal_.emit(record->key(),
                                                                                      Base::callingContext());
                    });
                  } catch (cms::Exception& caughtException) {
                    if (not excptr) {
                      edm::exceptionContext(caughtException, Base::callingContext());
                      excptr = std::current_exception();
                    }
                  }
                  if (excptr) {
                    Base::taskList().doneWaiting(excptr);
                    return;
                  }

                  Base::queue().push(
                      *group, [this, holder = std::move(holder), serviceToken, record, eventSetupImpl]() mutable {
                        Base::callingContext().setState(ESModuleCallingContext::State::kRunning);
                        std::exception_ptr exceptPtr;
                        try {
                          convertException::wrap([this, &holder, &serviceToken, &record, &eventSetupImpl] {
                            ESModuleCallingContext const& context = Base::callingContext();
                            auto proxies = Base::getTokenIndices();
                            if (Base::postMayGetResolvers()) {
                              proxies = &((*Base::postMayGetResolvers()).front());
                            }
                            TRecord rec;
                            edm::ESParentContext pc{&context};
                            rec.setImpl(record, Base::transitionID(), proxies, eventSetupImpl, &pc);
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
                            acquireCache_ = (*acquireFunction_)(rec, holder);
                          });
                        } catch (cms::Exception& iException) {
                          iException.addContext("Running acquire");
                          exceptPtr = std::current_exception();
                        }
                        holder.doneWaiting(exceptPtr);
                      });
                }));
      }

      WaitingTaskWithArenaHolder makeExceptionHandlerTask(WaitingTaskHolder produceTask,
                                                          oneapi::tbb::task_group* group) {
        return WaitingTaskWithArenaHolder(*group,
                                          make_waiting_task([this, produceTask = std::move(produceTask)](
                                                                std::exception_ptr const* iException) mutable {
                                            std::exception_ptr excptr;
                                            if (iException) {
                                              excptr = *iException;
                                            }
                                            if (excptr) {
                                              try {
                                                convertException::wrap([excptr]() { std::rethrow_exception(excptr); });
                                              } catch (cms::Exception& exception) {
                                                exception.addContext("Running acquire and external work");
                                                edm::exceptionContext(exception, Base::callingContext());
                                                produceTask.doneWaiting(std::current_exception());
                                              }
                                            }
                                          }));
      }

      std::shared_ptr<TAcquireFunc> acquireFunction_;
      typename impl::AcquireCacheType<TAcquireReturn>::type acquireCache_;
    };
  }  // namespace eventsetup
}  // namespace edm

#endif
