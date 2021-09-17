#ifndef FWCore_Framework_stream_EDAnalyzer_h
#define FWCore_Framework_stream_EDAnalyzer_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     EDAnalyzer
//
/**\class edm::stream::EDAnalyzer EDAnalyzer.h "FWCore/Framework/interface/stream/EDAnalyzer.h"

 Description: Base class for stream based EDAnalyzers

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu, 01 Aug 2013 21:41:42 GMT
//

#include "FWCore/Framework/interface/stream/AbilityToImplementor.h"
#include "FWCore/Framework/interface/stream/CacheContexts.h"
#include "FWCore/Framework/interface/stream/Contexts.h"
#include "FWCore/Framework/interface/stream/AbilityChecker.h"
#include "FWCore/Framework/interface/stream/EDAnalyzerBase.h"

namespace edm {
  namespace stream {

    template <typename... T>
    class EDAnalyzer : public AbilityToImplementor<T>::Type..., public EDAnalyzerBase {
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

      using EDAnalyzerBase::callWhenNewProductsRegistered;

      EDAnalyzer() = default;
      EDAnalyzer(const EDAnalyzer&) = delete;
      const EDAnalyzer& operator=(const EDAnalyzer&) = delete;
    };

  }  // namespace stream
}  // namespace edm

#endif
