#ifndef FWCore_Framework_one_analyzerAbilityToImplementor_h
#define FWCore_Framework_one_analyzerAbilityToImplementor_h
// -*- C++ -*-
//
// Package:     Package
// Class  :     analyzer::AbilityToImplementor
//
/**\class analyzer::AbilityToImplementor analyzerAbilityToImplementor.h "FWCore/Framework/interface/one/analyzerAbilityToImplementor.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu, 01 Aug 2013 19:39:58 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/moduleAbilities.h"
#include "FWCore/Framework/interface/one/moduleAbilities.h"
#include "FWCore/Framework/interface/one/implementors.h"
#include "FWCore/Framework/interface/one/EDAnalyzerBase.h"

// forward declarations

namespace edm {
  namespace one {
    namespace analyzer {
      template <typename T>
      struct AbilityToImplementor;

      template <>
      struct AbilityToImplementor<edm::one::SharedResources> {
        using Type = edm::one::impl::SharedResourcesUser<edm::one::EDAnalyzerBase>;
      };

      template <>
      struct AbilityToImplementor<edm::one::WatchRuns> {
        using Type = edm::one::impl::RunWatcher<edm::one::EDAnalyzerBase>;
      };

      template <>
      struct AbilityToImplementor<edm::one::WatchLuminosityBlocks> {
        using Type = edm::one::impl::LuminosityBlockWatcher<edm::one::EDAnalyzerBase>;
      };

      template <typename... Cs>
      struct AbilityToImplementor<edm::InputProcessBlockCache<Cs...>> {
        using Type = edm::one::impl::InputProcessBlockCacheHolder<edm::one::EDAnalyzerBase, Cs...>;
      };

      template <typename C>
      struct AbilityToImplementor<edm::RunCache<C>> {
        using Type = edm::one::impl::RunCacheHolder<edm::one::EDAnalyzerBase, C>;
      };

      template <typename C>
      struct AbilityToImplementor<edm::LuminosityBlockCache<C>> {
        using Type = edm::one::impl::LuminosityBlockCacheHolder<edm::one::EDAnalyzerBase, C>;
      };

      template <>
      struct AbilityToImplementor<edm::WatchProcessBlock> {
        using Type = edm::one::impl::WatchProcessBlock<edm::one::EDAnalyzerBase>;
      };
    }  // namespace analyzer
  }    // namespace one
}  // namespace edm

#endif
