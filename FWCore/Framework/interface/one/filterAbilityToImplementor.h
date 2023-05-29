#ifndef FWCore_Framework_one_filterAbilityToImplementor_h
#define FWCore_Framework_one_filterAbilityToImplementor_h
// -*- C++ -*-
//
// Package:     Package
// Class  :     filter::AbilityToImplementor
//
/**\class filter::AbilityToImplementor filterAbilityToImplementor.h "FWCore/Framework/interface/one/filterAbilityToImplementor.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu, 09 May 2013 19:39:58 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/moduleAbilities.h"
#include "FWCore/Framework/interface/one/moduleAbilities.h"
#include "FWCore/Framework/interface/one/implementors.h"
#include "FWCore/Framework/interface/one/EDFilterBase.h"

// forward declarations

namespace edm {
  namespace one {
    namespace filter {
      template <typename T>
      struct AbilityToImplementor;

      template <>
      struct AbilityToImplementor<edm::one::SharedResources> {
        using Type = edm::one::impl::SharedResourcesUser<edm::one::EDFilterBase>;
      };

      template <>
      struct AbilityToImplementor<edm::one::WatchRuns> {
        using Type = edm::one::impl::RunWatcher<edm::one::EDFilterBase>;
      };

      template <>
      struct AbilityToImplementor<edm::one::WatchLuminosityBlocks> {
        using Type = edm::one::impl::LuminosityBlockWatcher<edm::one::EDFilterBase>;
      };

      template <>
      struct AbilityToImplementor<edm::WatchProcessBlock> {
        using Type = edm::one::impl::WatchProcessBlock<edm::one::EDFilterBase>;
      };

      template <>
      struct AbilityToImplementor<edm::BeginProcessBlockProducer> {
        using Type = edm::one::impl::BeginProcessBlockProducer<edm::one::EDFilterBase>;
      };

      template <>
      struct AbilityToImplementor<edm::EndProcessBlockProducer> {
        using Type = edm::one::impl::EndProcessBlockProducer<edm::one::EDFilterBase>;
      };

      template <>
      struct AbilityToImplementor<edm::BeginRunProducer> {
        using Type = edm::one::impl::BeginRunProducer<edm::one::EDFilterBase>;
      };

      template <>
      struct AbilityToImplementor<edm::EndRunProducer> {
        using Type = edm::one::impl::EndRunProducer<edm::one::EDFilterBase>;
      };

      template <>
      struct AbilityToImplementor<edm::BeginLuminosityBlockProducer> {
        using Type = edm::one::impl::BeginLuminosityBlockProducer<edm::one::EDFilterBase>;
      };

      template <>
      struct AbilityToImplementor<edm::EndLuminosityBlockProducer> {
        using Type = edm::one::impl::EndLuminosityBlockProducer<edm::one::EDFilterBase>;
      };

      template <>
      struct AbilityToImplementor<edm::Transformer> {
        using Type = edm::one::impl::Transformer<edm::one::EDFilterBase>;
      };

      template <typename... Cs>
      struct AbilityToImplementor<edm::InputProcessBlockCache<Cs...>> {
        using Type = edm::one::impl::InputProcessBlockCacheHolder<edm::one::EDFilterBase, Cs...>;
      };

      template <typename C>
      struct AbilityToImplementor<edm::RunCache<C>> {
        using Type = edm::one::impl::RunCacheHolder<edm::one::EDFilterBase, C>;
      };

      template <typename C>
      struct AbilityToImplementor<edm::LuminosityBlockCache<C>> {
        using Type = edm::one::impl::LuminosityBlockCacheHolder<edm::one::EDFilterBase, C>;
      };

    }  // namespace filter
  }    // namespace one
}  // namespace edm

#endif
