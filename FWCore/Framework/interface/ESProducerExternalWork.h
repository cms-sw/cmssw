// -*- C++ -*-
#ifndef FWCore_Framework_ESProducerExternalWork_h
#define FWCore_Framework_ESProducerExternalWork_h
//
// Package:     Framework
// Class  :     ESProducerExternalWork
//
/**\class edm::ESProducer

 Description: Module to produce EventSetup data asynchronously in a manner
              similar to the ExternalWork feature of EDProducers.

 Usage: Same as ESProducer interface except there is the option to define
        a second "acquire" function if you use the setWhatAcquiredProduced
        function.
*/
//
// Author:      W. David Dagenhart
// Created:     27 February 2023

#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "FWCore/Concurrency/interface/WaitingTaskWithArenaHolder.h"
#include "FWCore/Framework/interface/Callback.h"
#include "FWCore/Framework/interface/CallbackExternalWork.h"
#include "FWCore/Framework/interface/ESConsumesCollector.h"
#include "FWCore/Framework/interface/es_Label.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/produce_helpers.h"

namespace edm {

  class ESProducerExternalWork : public ESProducer {
  public:
    ESProducerExternalWork();

    // These replicate the setWhatProduced functions but add a second functor for
    // the acquire step. The acquire step is analogous to the acquire step for
    // EDProducers with the ExternalWork ability.

    template <typename T>
    auto setWhatAcquiredProduced(T* iThis, const es::Label& iLabel = {}) {
      return setWhatAcquiredProduced(iThis, &T::acquire, &T::produce, iLabel);
    }

    template <typename T>
    auto setWhatAcquiredProduced(T* iThis, const char* iLabel) {
      return setWhatAcquiredProduced(iThis, es::Label(iLabel));
    }
    template <typename T>
    auto setWhatAcquiredProduced(T* iThis, const std::string& iLabel) {
      return setWhatAcquiredProduced(iThis, es::Label(iLabel));
    }

    // Note the Decorator pre and post functions are only run before
    // and after the 'produce' function, but they are not run before
    // and after the 'acquire' function.

    template <typename T, typename TDecorator>
    auto setWhatAcquiredProduced(T* iThis, const TDecorator& iDec, const es::Label& iLabel = {}) {
      return setWhatAcquiredProduced(iThis, &T::acquire, &T::produce, iDec, iLabel);
    }

    /** \param iThis the 'this' pointer to an inheriting class instance
        \param iAcquireMethod a member method of the inheriting class
        \param iProduceMethod a member method of the inheriting class
        The TRecord and TReturn template parameters can be deduced
        from iAquireMethod and iPRoduceMethod in order to do the
        registration with the EventSetup
    */
    template <typename T, typename TAcquireReturn, typename TProduceReturn, typename TRecord>
    auto setWhatAcquiredProduced(T* iThis,
                                 TAcquireReturn (T::*iAcquireMethod)(const TRecord&, WaitingTaskWithArenaHolder),
                                 TProduceReturn (T::*iProduceMethod)(const TRecord&, TAcquireReturn),
                                 const es::Label& iLabel = {}) {
      return setWhatAcquiredProduced(
          iThis, iAcquireMethod, iProduceMethod, eventsetup::CallbackSimpleDecorator<TRecord>(), iLabel);
    }

    template <typename T, typename TAcquireReturn, typename TProduceReturn, typename TRecord, typename TDecorator>
    auto setWhatAcquiredProduced(T* iThis,
                                 TAcquireReturn (T::*iAcquireMethod)(const TRecord&, WaitingTaskWithArenaHolder),
                                 TProduceReturn (T ::*iProduceMethod)(const TRecord&, TAcquireReturn),
                                 const TDecorator& iDec,
                                 const es::Label& iLabel = {}) {
      return setWhatAcquiredProducedWithLambda<TAcquireReturn, TProduceReturn, TRecord>(
          [iThis, iAcquireMethod](TRecord const& iRecord, WaitingTaskWithArenaHolder iHolder) {
            return (iThis->*iAcquireMethod)(iRecord, std::move(iHolder));
          },
          [iThis, iProduceMethod](TRecord const& iRecord, TAcquireReturn iAcquireReturn) {
            return (iThis->*iProduceMethod)(iRecord, std::move(iAcquireReturn));
          },
          createDecoratorFrom(iThis, static_cast<const TRecord*>(nullptr), iDec),
          iLabel);
    }

    /**
     * This overload allows lambdas (functors) to be used as the
     * production function. As of now it is not intended for wide use
     * (we are thinking for a better API for users)
     */
    template <typename TAcquireFunc, typename TProduceFunc>
    auto setWhatAcquiredProducedWithLambda(TAcquireFunc&& acquireFunc,
                                           TProduceFunc&& produceFunc,
                                           const es::Label& iLabel = {}) {
      using AcquireTypes = eventsetup::impl::ReturnArgumentTypes<TAcquireFunc>;
      using TAcquireReturn = typename AcquireTypes::return_type;
      using ProduceTypes = eventsetup::impl::ReturnArgumentTypes<TProduceFunc>;
      using TProduceReturn = typename ProduceTypes::return_type;
      using TRecord = typename ProduceTypes::argument_type;
      using DecoratorType = eventsetup::CallbackSimpleDecorator<TRecord>;

      return setWhatAcquiredProducedWithLambda<TAcquireReturn, TProduceReturn, TRecord>(
          std::forward<TAcquireFunc>(acquireFunc), std::forward<TProduceFunc>(produceFunc), DecoratorType(), iLabel);
    }

    // In this template, TReturn and TRecord cannot be deduced. They must be explicitly provided when called.
    // The previous 7 functions all end up calling this one (directly or indirectly).
    template <typename TAcquireReturn,
              typename TProduceReturn,
              typename TRecord,
              typename TAcquireFunc,
              typename TProduceFunc,
              typename TDecorator>
    ESConsumesCollectorT<TRecord> setWhatAcquiredProducedWithLambda(TAcquireFunc&& acquireFunc,
                                                                    TProduceFunc&& produceFunc,
                                                                    TDecorator&& iDec,
                                                                    const es::Label& iLabel = {}) {
      const auto id = consumesInfoSize();
      using DecoratorType = std::decay_t<TDecorator>;
      using CallbackType = eventsetup::CallbackExternalWork<ESProducerExternalWork,
                                                            TAcquireFunc,
                                                            TAcquireReturn,
                                                            TProduceFunc,
                                                            TProduceReturn,
                                                            TRecord,
                                                            DecoratorType>;
      unsigned int iovIndex = 0;  // Start with 0, but later will cycle through all of them
      auto temp = std::make_shared<CallbackType>(this,
                                                 std::forward<TAcquireFunc>(acquireFunc),
                                                 std::forward<TProduceFunc>(produceFunc),
                                                 id,
                                                 std::forward<TDecorator>(iDec));
      auto callback =
          std::make_shared<std::pair<unsigned int, std::shared_ptr<CallbackType>>>(iovIndex, std::move(temp));
      registerProducts(std::move(callback),
                       static_cast<const typename eventsetup::produce::product_traits<TProduceReturn>::type*>(nullptr),
                       static_cast<const TRecord*>(nullptr),
                       iLabel);
      return ESConsumesCollectorT<TRecord>(consumesInfoPushBackNew(), id);
    }
  };
}  // namespace edm
#endif
