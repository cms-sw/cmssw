#ifndef FWCore_Framework_limited_analyzerAbilityToImplementor_h
#define FWCore_Framework_limited_analyzerAbilityToImplementor_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// File  :     analyzerAbilityToImplementor
//
/**\file  analyzerAbilityToImplementor.h "FWCore/Framework/interface/limited/analyzerAbilityToImplementor.h"

 Description: Class used to pair a module Ability to the actual base class used to implement that ability

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu, 18 Jul 2013 11:51:33 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/moduleAbilities.h"
#include "FWCore/Framework/interface/limited/implementors.h"
#include "FWCore/Framework/interface/limited/EDAnalyzerBase.h"

// forward declarations
namespace edm {
  namespace limited {
    namespace analyzer {
      template <typename T>
      struct AbilityToImplementor;

      template <typename C>
      struct AbilityToImplementor<edm::StreamCache<C>> {
        using Type = edm::limited::impl::StreamCacheHolder<edm::limited::EDAnalyzerBase, C>;
      };

      template <typename... Cs>
      struct AbilityToImplementor<edm::InputProcessBlockCache<Cs...>> {
        using Type = edm::limited::impl::InputProcessBlockCacheHolder<edm::limited::EDAnalyzerBase, Cs...>;
      };

      template <typename C>
      struct AbilityToImplementor<edm::RunCache<C>> {
        using Type = edm::limited::impl::RunCacheHolder<edm::limited::EDAnalyzerBase, C>;
      };

      template <typename C>
      struct AbilityToImplementor<edm::RunSummaryCache<C>> {
        using Type = edm::limited::impl::RunSummaryCacheHolder<edm::limited::EDAnalyzerBase, C>;
      };

      template <typename C>
      struct AbilityToImplementor<edm::LuminosityBlockCache<C>> {
        using Type = edm::limited::impl::LuminosityBlockCacheHolder<edm::limited::EDAnalyzerBase, C>;
      };

      template <typename C>
      struct AbilityToImplementor<edm::LuminosityBlockSummaryCache<C>> {
        using Type = edm::limited::impl::LuminosityBlockSummaryCacheHolder<edm::limited::EDAnalyzerBase, C>;
      };

      template <>
      struct AbilityToImplementor<edm::WatchProcessBlock> {
        using Type = edm::limited::impl::WatchProcessBlock<edm::limited::EDAnalyzerBase>;
      };
    }  // namespace analyzer
  }    // namespace limited
}  // namespace edm

#endif
