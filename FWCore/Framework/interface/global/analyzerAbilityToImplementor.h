#ifndef FWCore_Framework_global_analyzerAbilityToImplementor_h
#define FWCore_Framework_global_analyzerAbilityToImplementor_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// File  :     analyzerAbilityToImplementor
//
/**\file  analyzerAbilityToImplementor.h "FWCore/Framework/interface/global/analyzerAbilityToImplementor.h"

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
#include "FWCore/Framework/interface/global/implementors.h"
#include "FWCore/Framework/interface/global/EDAnalyzerBase.h"

// forward declarations
namespace edm {
  namespace global {
    namespace analyzer {
      template <typename T>
      struct AbilityToImplementor;

      template <typename C>
      struct AbilityToImplementor<edm::StreamCache<C>> {
        using Type = edm::global::impl::StreamCacheHolder<edm::global::EDAnalyzerBase, C>;
      };

      template <typename... Cs>
      struct AbilityToImplementor<edm::InputProcessBlockCache<Cs...>> {
        using Type = edm::global::impl::InputProcessBlockCacheHolder<edm::global::EDAnalyzerBase, Cs...>;
      };

      template <typename C>
      struct AbilityToImplementor<edm::RunCache<C>> {
        using Type = edm::global::impl::RunCacheHolder<edm::global::EDAnalyzerBase, C>;
      };

      template <typename C>
      struct AbilityToImplementor<edm::RunSummaryCache<C>> {
        using Type = edm::global::impl::RunSummaryCacheHolder<edm::global::EDAnalyzerBase, C>;
      };

      template <typename C>
      struct AbilityToImplementor<edm::LuminosityBlockCache<C>> {
        using Type = edm::global::impl::LuminosityBlockCacheHolder<edm::global::EDAnalyzerBase, C>;
      };

      template <typename C>
      struct AbilityToImplementor<edm::LuminosityBlockSummaryCache<C>> {
        using Type = edm::global::impl::LuminosityBlockSummaryCacheHolder<edm::global::EDAnalyzerBase, C>;
      };

      template <>
      struct AbilityToImplementor<edm::WatchProcessBlock> {
        using Type = edm::global::impl::WatchProcessBlock<edm::global::EDAnalyzerBase>;
      };
    }  // namespace analyzer
  }    // namespace global
}  // namespace edm

#endif
