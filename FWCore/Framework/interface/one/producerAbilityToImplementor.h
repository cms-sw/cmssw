#ifndef FWCore_Framework_one_producerAbilityToImplementor_h
#define FWCore_Framework_one_producerAbilityToImplementor_h
// -*- C++ -*-
//
// Package:     Package
// Class  :     producer::AbilityToImplementor
//
/**\class producer::AbilityToImplementor producerAbilityToImplementor.h "FWCore/Framework/interface/one/producerAbilityToImplementor.h"

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
#include "FWCore/Framework/interface/one/EDProducerBase.h"

// forward declarations

namespace edm {
  namespace one {
    namespace producer {
      template <typename T>
      struct AbilityToImplementor;

      template <>
      struct AbilityToImplementor<edm::one::SharedResources> {
        using Type = edm::one::impl::SharedResourcesUser<edm::one::EDProducerBase>;
      };

      template <>
      struct AbilityToImplementor<edm::one::WatchRuns> {
        using Type = edm::one::impl::RunWatcher<edm::one::EDProducerBase>;
      };

      template <>
      struct AbilityToImplementor<edm::one::WatchLuminosityBlocks> {
        using Type = edm::one::impl::LuminosityBlockWatcher<edm::one::EDProducerBase>;
      };

      template <>
      struct AbilityToImplementor<edm::WatchProcessBlock> {
        using Type = edm::one::impl::WatchProcessBlock<edm::one::EDProducerBase>;
      };

      template <>
      struct AbilityToImplementor<edm::BeginProcessBlockProducer> {
        using Type = edm::one::impl::BeginProcessBlockProducer<edm::one::EDProducerBase>;
      };

      template <>
      struct AbilityToImplementor<edm::EndProcessBlockProducer> {
        using Type = edm::one::impl::EndProcessBlockProducer<edm::one::EDProducerBase>;
      };

      template <>
      struct AbilityToImplementor<edm::BeginRunProducer> {
        using Type = edm::one::impl::BeginRunProducer<edm::one::EDProducerBase>;
      };

      template <>
      struct AbilityToImplementor<edm::EndRunProducer> {
        using Type = edm::one::impl::EndRunProducer<edm::one::EDProducerBase>;
      };

      template <>
      struct AbilityToImplementor<edm::BeginLuminosityBlockProducer> {
        using Type = edm::one::impl::BeginLuminosityBlockProducer<edm::one::EDProducerBase>;
      };

      template <>
      struct AbilityToImplementor<edm::EndLuminosityBlockProducer> {
        using Type = edm::one::impl::EndLuminosityBlockProducer<edm::one::EDProducerBase>;
      };

      template <>
      struct AbilityToImplementor<edm::Transformer> {
        using Type = edm::one::impl::Transformer<edm::one::EDProducerBase>;
      };

      template <typename... Cs>
      struct AbilityToImplementor<edm::InputProcessBlockCache<Cs...>> {
        using Type = edm::one::impl::InputProcessBlockCacheHolder<edm::one::EDProducerBase, Cs...>;
      };

      template <typename C>
      struct AbilityToImplementor<edm::RunCache<C>> {
        using Type = edm::one::impl::RunCacheHolder<edm::one::EDProducerBase, C>;
      };

      template <typename C>
      struct AbilityToImplementor<edm::LuminosityBlockCache<C>> {
        using Type = edm::one::impl::LuminosityBlockCacheHolder<edm::one::EDProducerBase, C>;
      };

      template <>
      struct AbilityToImplementor<edm::Accumulator> {
        using Type = edm::one::impl::Accumulator<edm::one::EDProducerBase>;
      };

    }  // namespace producer
  }    // namespace one
}  // namespace edm

#endif
