#ifndef FWCore_Framework_stream_EDFilter_h
#define FWCore_Framework_stream_EDFilter_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     EDFilter
//
/**\class edm::stream::EDFilter EDFilter.h "FWCore/Framework/interface/stream/EDFilter.h"

 Description: Base class for stream based EDFilters

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu, 01 Aug 2013 21:41:42 GMT
//

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/AbilityToImplementor.h"
#include "FWCore/Framework/interface/stream/CacheContexts.h"
#include "FWCore/Framework/interface/stream/Contexts.h"
#include "FWCore/Framework/interface/stream/AbilityChecker.h"
#include "FWCore/Framework/interface/stream/EDFilterBase.h"
#include "FWCore/Framework/interface/stream/ProducingModuleHelper.h"

namespace edm {

  class WaitingTaskWithArenaHolder;

  namespace stream {

    template <typename... T>
    class EDFilter : public AbilityToImplementor<T>::Type..., public EDFilterBase {
    public:
      using CacheTypes = CacheContexts<T...>;

      using GlobalCache = typename CacheTypes::GlobalCache;
      using InputProcessBlockCache = typename CacheTypes::InputProcessBlockCache;
      using RunCache = typename CacheTypes::RunCache;
      using LuminosityBlockCache = typename CacheTypes::LuminosityBlockCache;
      using RunContext = RunContextT<RunCache, GlobalCache>;
      using LuminosityBlockContext = LuminosityBlockContextT<LuminosityBlockCache, RunCache, GlobalCache>;
      using RunSummaryCache = typename CacheTypes::RunSummaryCache;
      using LuminosityBlockSummaryCache = typename CacheTypes::LuminosityBlockSummaryCache;

      using HasAbility = AbilityChecker<T...>;

      EDFilter() = default;
      EDFilter(const EDFilter&) = delete;
      const EDFilter& operator=(const EDFilter&) = delete;

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

    private:
      void doAcquire_(Event const& ev, EventSetup const& es, WaitingTaskWithArenaHolder& holder) final {
        doAcquireIfNeeded(this, ev, es, holder);
      }
    };

  }  // namespace stream
}  // namespace edm

#endif
