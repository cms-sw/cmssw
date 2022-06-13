#ifndef FWCore_Framework_limited_producerAbilityToImplementor_h
#define FWCore_Framework_limited_producerAbilityToImplementor_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// File  :     producerAbilityToImplementor
//
/**\file  producerAbilityToImplementor.h "FWCore/Framework/interface/limited/producerAbilityToImplementor.h"

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
#include "FWCore/Framework/interface/limited/EDProducerBase.h"

// forward declarations
namespace edm {
  namespace limited {
    namespace producer {
      template <typename T>
      struct AbilityToImplementor;

      template <typename C>
      struct AbilityToImplementor<edm::StreamCache<C>> {
        using Type = edm::limited::impl::StreamCacheHolder<edm::limited::EDProducerBase, C>;
      };

      template <typename... Cs>
      struct AbilityToImplementor<edm::InputProcessBlockCache<Cs...>> {
        using Type = edm::limited::impl::InputProcessBlockCacheHolder<edm::limited::EDProducerBase, Cs...>;
      };

      template <typename C>
      struct AbilityToImplementor<edm::RunCache<C>> {
        using Type = edm::limited::impl::RunCacheHolder<edm::limited::EDProducerBase, C>;
      };

      template <typename C>
      struct AbilityToImplementor<edm::RunSummaryCache<C>> {
        using Type = edm::limited::impl::RunSummaryCacheHolder<edm::limited::EDProducerBase, C>;
      };

      template <typename C>
      struct AbilityToImplementor<edm::LuminosityBlockCache<C>> {
        using Type = edm::limited::impl::LuminosityBlockCacheHolder<edm::limited::EDProducerBase, C>;
      };

      template <typename C>
      struct AbilityToImplementor<edm::LuminosityBlockSummaryCache<C>> {
        using Type = edm::limited::impl::LuminosityBlockSummaryCacheHolder<edm::limited::EDProducerBase, C>;
      };

      template <>
      struct AbilityToImplementor<edm::WatchProcessBlock> {
        using Type = edm::limited::impl::WatchProcessBlock<edm::limited::EDProducerBase>;
      };

      template <>
      struct AbilityToImplementor<edm::BeginProcessBlockProducer> {
        using Type = edm::limited::impl::BeginProcessBlockProducer<edm::limited::EDProducerBase>;
      };

      template <>
      struct AbilityToImplementor<edm::EndProcessBlockProducer> {
        using Type = edm::limited::impl::EndProcessBlockProducer<edm::limited::EDProducerBase>;
      };

      template <>
      struct AbilityToImplementor<edm::BeginRunProducer> {
        using Type = edm::limited::impl::BeginRunProducer<edm::limited::EDProducerBase>;
      };

      template <>
      struct AbilityToImplementor<edm::EndRunProducer> {
        using Type = edm::limited::impl::EndRunProducer<edm::limited::EDProducerBase>;
      };

      template <>
      struct AbilityToImplementor<edm::BeginLuminosityBlockProducer> {
        using Type = edm::limited::impl::BeginLuminosityBlockProducer<edm::limited::EDProducerBase>;
      };

      template <>
      struct AbilityToImplementor<edm::EndLuminosityBlockProducer> {
        using Type = edm::limited::impl::EndLuminosityBlockProducer<edm::limited::EDProducerBase>;
      };

      template <>
      struct AbilityToImplementor<edm::Transformer> {
        using Type = edm::limited::impl::Transformer<edm::limited::EDProducerBase>;
      };

      template <>
      struct AbilityToImplementor<edm::Accumulator> {
        using Type = edm::limited::impl::Accumulator<edm::limited::EDProducerBase>;
      };

      template <bool, bool, typename T>
      struct SpecializeAbilityToImplementor {
        using Type = typename AbilityToImplementor<T>::Type;
      };

      template <bool B, typename C>
      struct SpecializeAbilityToImplementor<true, B, edm::RunSummaryCache<C>> {
        using Type = typename edm::limited::impl::EndRunSummaryProducer<edm::limited::EDProducerBase, C>;
      };

      template <bool B>
      struct SpecializeAbilityToImplementor<true, B, edm::EndRunProducer> {
        using Type = edm::limited::impl::EmptyType;
      };

      template <bool B, typename C>
      struct SpecializeAbilityToImplementor<B, true, edm::LuminosityBlockSummaryCache<C>> {
        using Type = typename edm::limited::impl::EndLuminosityBlockSummaryProducer<edm::limited::EDProducerBase, C>;
      };

      template <bool B>
      struct SpecializeAbilityToImplementor<B, true, edm::EndLuminosityBlockProducer> {
        using Type = edm::limited::impl::EmptyType;
      };
    }  // namespace producer
  }    // namespace limited
}  // namespace edm

#endif
