#ifndef FWCore_Framework_limited_EDFilter_h
#define FWCore_Framework_limited_EDFilter_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     edm::limited::EDFilter
//
/**\class edm::limited::EDFilter EDFilter.h "FWCore/Framework/interface/limited/EDFilter.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Tue, 23 Jul 2013 11:51:07 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/limited/filterAbilityToImplementor.h"
#include "FWCore/Framework/interface/moduleAbilities.h"

// forward declarations

namespace edm {
  namespace limited {
    template <typename... T>
    class EDFilter : public virtual EDFilterBase,
                     public filter::SpecializeAbilityToImplementor<
                         CheckAbility<edm::module::Abilities::kRunSummaryCache, T...>::kHasIt &
                             CheckAbility<edm::module::Abilities::kEndRunProducer, T...>::kHasIt,
                         CheckAbility<edm::module::Abilities::kLuminosityBlockSummaryCache, T...>::kHasIt &
                             CheckAbility<edm::module::Abilities::kEndLuminosityBlockProducer, T...>::kHasIt,
                         T>::Type... {
    public:
      EDFilter(edm::ParameterSet const& iPSet)
          : EDFilterBase(iPSet),
            filter::SpecializeAbilityToImplementor<
                CheckAbility<edm::module::Abilities::kRunSummaryCache, T...>::kHasIt &
                    CheckAbility<edm::module::Abilities::kEndRunProducer, T...>::kHasIt,
                CheckAbility<edm::module::Abilities::kLuminosityBlockSummaryCache, T...>::kHasIt &
                    CheckAbility<edm::module::Abilities::kEndLuminosityBlockProducer, T...>::kHasIt,
                T>::Type(iPSet)... {}
// We do this only in the case of the intel compiler as this might
// end up creating a lot of code bloat due to inline symbols being generated
// in each DSO which uses this header.
#ifdef __INTEL_COMPILER
      virtual ~EDFilter() = default;
#endif
      // ---------- const member functions ---------------------
      bool wantsProcessBlocks() const final { return WantsProcessBlockTransitions<T...>::value; }
      bool wantsInputProcessBlocks() const final { return WantsInputProcessBlockTransitions<T...>::value; }
      bool wantsGlobalRuns() const final { return WantsGlobalRunTransitions<T...>::value; }
      bool wantsGlobalLuminosityBlocks() const final { return WantsGlobalLuminosityBlockTransitions<T...>::value; }
      bool wantsStreamRuns() const final { return WantsStreamRunTransitions<T...>::value; }
      bool wantsStreamLuminosityBlocks() const final { return WantsStreamLuminosityBlockTransitions<T...>::value; }

      bool hasAbilityToProduceInBeginProcessBlocks() const final {
        return HasAbilityToProduceInBeginProcessBlocks<T...>::value;
      }
      bool hasAbilityToProduceInEndProcessBlocks() const final {
        return HasAbilityToProduceInEndProcessBlocks<T...>::value;
      }

      bool hasAbilityToProduceInBeginRuns() const final { return HasAbilityToProduceInBeginRuns<T...>::value; }
      bool hasAbilityToProduceInEndRuns() const final { return HasAbilityToProduceInEndRuns<T...>::value; }

      bool hasAbilityToProduceInBeginLumis() const final { return HasAbilityToProduceInBeginLumis<T...>::value; }
      bool hasAbilityToProduceInEndLumis() const final { return HasAbilityToProduceInEndLumis<T...>::value; }

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

    private:
      EDFilter(const EDFilter&) = delete;

      const EDFilter& operator=(const EDFilter&) = delete;

      // ---------- member data --------------------------------
    };

  }  // namespace limited
}  // namespace edm

#endif
