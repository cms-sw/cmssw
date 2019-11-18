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

// system include files

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/AbilityToImplementor.h"
#include "FWCore/Framework/interface/stream/CacheContexts.h"
#include "FWCore/Framework/interface/stream/Contexts.h"
#include "FWCore/Framework/interface/stream/AbilityChecker.h"
#include "FWCore/Framework/interface/stream/EDFilterBase.h"
#include "FWCore/Framework/interface/stream/ProducingModuleHelper.h"
// forward declarations
namespace edm {

  class WaitingTaskWithArenaHolder;

  namespace stream {
    template <typename... T>
    class EDFilter : public AbilityToImplementor<T>::Type..., public EDFilterBase {
    public:
      typedef CacheContexts<T...> CacheTypes;

      typedef typename CacheTypes::GlobalCache GlobalCache;
      typedef typename CacheTypes::RunCache RunCache;
      typedef typename CacheTypes::LuminosityBlockCache LuminosityBlockCache;
      typedef RunContextT<RunCache, GlobalCache> RunContext;
      typedef LuminosityBlockContextT<LuminosityBlockCache, RunCache, GlobalCache> LuminosityBlockContext;
      typedef typename CacheTypes::RunSummaryCache RunSummaryCache;
      typedef typename CacheTypes::LuminosityBlockSummaryCache LuminosityBlockSummaryCache;

      typedef AbilityChecker<T...> HasAbility;

      EDFilter() = default;
      //virtual ~EDFilter();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

      bool hasAbilityToProduceInBeginRuns() const final { return HasAbilityToProduceInBeginRuns<T...>::value; }
      bool hasAbilityToProduceInEndRuns() const final { return HasAbilityToProduceInEndRuns<T...>::value; }

      bool hasAbilityToProduceInBeginLumis() const final { return HasAbilityToProduceInBeginLumis<T...>::value; }
      bool hasAbilityToProduceInEndLumis() const final { return HasAbilityToProduceInEndLumis<T...>::value; }

    private:
      EDFilter(const EDFilter&) = delete;  // stop default

      const EDFilter& operator=(const EDFilter&) = delete;  // stop default

      void doAcquire_(Event const& ev, EventSetup const& es, WaitingTaskWithArenaHolder& holder) final {
        doAcquireIfNeeded(this, ev, es, holder);
      }

      // ---------- member data --------------------------------
    };

  }  // namespace stream
}  // namespace edm

#endif
