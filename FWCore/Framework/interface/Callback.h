// -*- C++ -*-
#ifndef FWCore_Framework_Callback_h
#define FWCore_Framework_Callback_h
//
// Package:     Framework
// Class  :     Callback
//
/**\class edm::eventsetup::Callback

 Description: Functional object used as the 'callback' for the CallbackProxy

 Usage: Produces data objects for ESProducers in EventSetup system

*/
//
// Author:      Chris Jones
// Created:     Sun Apr 17 14:30:24 EDT 2005
//

#include <memory>
#include <utility>

#include "oneapi/tbb/task_group.h"

#include "FWCore/Concurrency/interface/WaitingTaskHolder.h"
#include "FWCore/Framework/interface/CallbackBase.h"
#include "FWCore/ServiceRegistry/interface/ESParentContext.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"

namespace edm {

  class EventSetupImpl;

  namespace eventsetup {

    class EventSetupRecordImpl;

    template <typename T,             //producer's type
              typename TProduceFunc,  //produce functor type
              typename TReturn,       //return type of the produce method
              typename TRecord,       //the record passed in as an argument
              typename TDecorator     //allows customization using pre/post calls
              = CallbackSimpleDecorator<TRecord>>
    class Callback : public CallbackBase<T, TProduceFunc, TReturn, TRecord, TDecorator> {
    public:
      using Base = CallbackBase<T, TProduceFunc, TReturn, TRecord, TDecorator>;

      Callback(T* iProd, TProduceFunc iProduceFunc, unsigned int iID, const TDecorator& iDec = TDecorator())
          : Callback(iProd, std::make_shared<TProduceFunc>(std::move(iProduceFunc)), iID, iDec) {}

      Callback* clone() {
        return new Callback(Base::producer(), Base::produceFunction(), Base::transitionID(), Base::decorator());
      }

      void prefetchAsync(WaitingTaskHolder iTask,
                         EventSetupRecordImpl const* iRecord,
                         EventSetupImpl const* iEventSetupImpl,
                         ServiceToken const& token,
                         ESParentContext const& iParent) {
        return Base::prefetchAsyncImpl(
            [this](auto&& group, auto&& token, auto&& record, auto&& es) {
              constexpr bool emitPostPrefetchingSignal = true;
              auto produceFunctor = [this](TRecord const& record) { return (*Base::produceFunction())(record); };
              return Base::makeProduceTask(
                  group, token, record, es, emitPostPrefetchingSignal, std::move(produceFunctor));
            },
            std::move(iTask),
            iRecord,
            iEventSetupImpl,
            token,
            iParent);
      }

    private:
      Callback(T* iProd,
               std::shared_ptr<TProduceFunc> iProduceFunc,
               unsigned int iID,
               const TDecorator& iDec = TDecorator())
          : Base(iProd, std::move(iProduceFunc), iID, iDec) {}
    };
  }  // namespace eventsetup
}  // namespace edm
#endif
