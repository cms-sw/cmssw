#ifndef FWCore_Framework_one_EDProducer_h
#define FWCore_Framework_one_EDProducer_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     edm::one::EDProducer
//
/**\class edm::one::EDProducer EDProducer.h "FWCore/Framework/interface/one/EDProducer.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu, 09 May 2013 19:53:55 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/one/producerAbilityToImplementor.h"

// forward declarations
namespace edm {
  namespace one {
    template <typename... T>
    class EDProducer : public virtual EDProducerBase, public producer::AbilityToImplementor<T>::Type... {
    public:
      static_assert(not(CheckAbility<module::Abilities::kRunCache, T...>::kHasIt and
                        CheckAbility<module::Abilities::kOneWatchRuns, T...>::kHasIt),
                    "Cannot use both WatchRuns and RunCache");
      static_assert(not(CheckAbility<module::Abilities::kLuminosityBlockCache, T...>::kHasIt and
                        CheckAbility<module::Abilities::kOneWatchLuminosityBlocks, T...>::kHasIt),
                    "Cannot use both WatchLuminosityBlocks and LuminosityBLockCache");

      EDProducer() = default;
#ifdef __INTEL_COMPILER
      virtual ~EDProducer() = default;
#endif
      //

      // ---------- const member functions ---------------------
      bool wantsGlobalRuns() const final { return WantsGlobalRunTransitions<T...>::value; }
      bool wantsGlobalLuminosityBlocks() const final { return WantsGlobalLuminosityBlockTransitions<T...>::value; }

      bool hasAbilityToProduceInBeginRuns() const final { return HasAbilityToProduceInBeginRuns<T...>::value; }
      bool hasAbilityToProduceInEndRuns() const final { return HasAbilityToProduceInEndRuns<T...>::value; }

      bool hasAbilityToProduceInBeginLumis() const final { return HasAbilityToProduceInBeginLumis<T...>::value; }
      bool hasAbilityToProduceInEndLumis() const final { return HasAbilityToProduceInEndLumis<T...>::value; }

      SerialTaskQueue* globalRunsQueue() final { return globalRunsQueue_.queue(); }
      SerialTaskQueue* globalLuminosityBlocksQueue() final { return globalLuminosityBlocksQueue_.queue(); }

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

    private:
      EDProducer(const EDProducer&) = delete;
      const EDProducer& operator=(const EDProducer&) = delete;

      // ---------- member data --------------------------------
      impl::OptionalSerialTaskQueueHolder<WantsSerialGlobalRunTransitions<T...>::value> globalRunsQueue_;
      impl::OptionalSerialTaskQueueHolder<WantsSerialGlobalLuminosityBlockTransitions<T...>::value>
          globalLuminosityBlocksQueue_;
    };

  }  // namespace one
}  // namespace edm

#endif
