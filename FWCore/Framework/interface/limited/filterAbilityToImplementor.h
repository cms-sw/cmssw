#ifndef FWCore_Framework_limited_filterAbilityToImplementor_h
#define FWCore_Framework_limited_filterAbilityToImplementor_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// File  :     filterAbilityToImplementor
//
/**\file  filterAbilityToImplementor.h "FWCore/Framework/interface/limited/filterAbilityToImplementor.h"

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
#include "FWCore/Framework/interface/limited/EDFilterBase.h"

// forward declarations
namespace edm {
  namespace limited {
    namespace filter {
      template <typename T>
      struct AbilityToImplementor;

      template <typename C>
      struct AbilityToImplementor<edm::StreamCache<C>> {
        using Type = edm::limited::impl::StreamCacheHolder<edm::limited::EDFilterBase, C>;
      };

      template <typename... Cs>
      struct AbilityToImplementor<edm::InputProcessBlockCache<Cs...>> {
        using Type = edm::limited::impl::InputProcessBlockCacheHolder<edm::limited::EDFilterBase, Cs...>;
      };

      template <typename C>
      struct AbilityToImplementor<edm::RunCache<C>> {
        using Type = edm::limited::impl::RunCacheHolder<edm::limited::EDFilterBase, C>;
      };

      template <typename C>
      struct AbilityToImplementor<edm::RunSummaryCache<C>> {
        using Type = edm::limited::impl::RunSummaryCacheHolder<edm::limited::EDFilterBase, C>;
      };

      template <typename C>
      struct AbilityToImplementor<edm::LuminosityBlockCache<C>> {
        using Type = edm::limited::impl::LuminosityBlockCacheHolder<edm::limited::EDFilterBase, C>;
      };

      template <typename C>
      struct AbilityToImplementor<edm::LuminosityBlockSummaryCache<C>> {
        using Type = edm::limited::impl::LuminosityBlockSummaryCacheHolder<edm::limited::EDFilterBase, C>;
      };

      template <>
      struct AbilityToImplementor<edm::WatchProcessBlock> {
        using Type = edm::limited::impl::WatchProcessBlock<edm::limited::EDFilterBase>;
      };

      template <>
      struct AbilityToImplementor<edm::BeginProcessBlockProducer> {
        using Type = edm::limited::impl::BeginProcessBlockProducer<edm::limited::EDFilterBase>;
      };

      template <>
      struct AbilityToImplementor<edm::EndProcessBlockProducer> {
        using Type = edm::limited::impl::EndProcessBlockProducer<edm::limited::EDFilterBase>;
      };

      template <>
      struct AbilityToImplementor<edm::BeginRunProducer> {
        using Type = edm::limited::impl::BeginRunProducer<edm::limited::EDFilterBase>;
      };

      template <>
      struct AbilityToImplementor<edm::EndRunProducer> {
        using Type = edm::limited::impl::EndRunProducer<edm::limited::EDFilterBase>;
      };

      template <>
      struct AbilityToImplementor<edm::BeginLuminosityBlockProducer> {
        using Type = edm::limited::impl::BeginLuminosityBlockProducer<edm::limited::EDFilterBase>;
      };

      template <>
      struct AbilityToImplementor<edm::EndLuminosityBlockProducer> {
        using Type = edm::limited::impl::EndLuminosityBlockProducer<edm::limited::EDFilterBase>;
      };

      template <>
      struct AbilityToImplementor<edm::Transformer> {
        using Type = edm::limited::impl::Transformer<edm::limited::EDFilterBase>;
      };

      template <bool, bool, typename T>
      struct SpecializeAbilityToImplementor {
        using Type = typename AbilityToImplementor<T>::Type;
      };

      template <bool B, typename C>
      struct SpecializeAbilityToImplementor<true, B, edm::RunSummaryCache<C>> {
        using Type = typename edm::limited::impl::EndRunSummaryProducer<edm::limited::EDFilterBase, C>;
      };

      template <bool B>
      struct SpecializeAbilityToImplementor<true, B, edm::EndRunProducer> {
        using Type = edm::limited::impl::EmptyType;
      };

      template <bool B, typename C>
      struct SpecializeAbilityToImplementor<B, true, edm::LuminosityBlockSummaryCache<C>> {
        using Type = typename edm::limited::impl::EndLuminosityBlockSummaryProducer<edm::limited::EDFilterBase, C>;
      };

      template <bool B>
      struct SpecializeAbilityToImplementor<B, true, edm::EndLuminosityBlockProducer> {
        using Type = edm::limited::impl::EmptyType;
      };
    }  // namespace filter
  }    // namespace limited
}  // namespace edm

#endif
