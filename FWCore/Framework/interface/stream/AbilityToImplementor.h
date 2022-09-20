#ifndef FWCore_Framework_stream_AbilityToImplementor_h
#define FWCore_Framework_stream_AbilityToImplementor_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// File  :     AbilityToImplementor
//
/**\class  edm::stream::AbilityToImplementor producerAbilityToImplementor.h "FWCore/Framework/interface/stream/AbilityToImplementor.h"

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
#include "FWCore/Framework/interface/stream/implementors.h"

// forward declarations
namespace edm {
  namespace stream {
    template <typename T>
    struct AbilityToImplementor;

    template <typename C>
    struct AbilityToImplementor<edm::GlobalCache<C>> {
      using Type = edm::stream::impl::GlobalCacheHolder<C>;
    };

    template <typename... CacheTypes>
    struct AbilityToImplementor<edm::InputProcessBlockCache<CacheTypes...>> {
      using Type = edm::stream::impl::InputProcessBlockCacheHolder<CacheTypes...>;
    };

    template <typename C>
    struct AbilityToImplementor<edm::RunCache<C>> {
      using Type = edm::stream::impl::RunCacheHolder<C>;
    };

    template <typename C>
    struct AbilityToImplementor<edm::RunSummaryCache<C>> {
      using Type = edm::stream::impl::RunSummaryCacheHolder<C>;
    };

    template <typename C>
    struct AbilityToImplementor<edm::LuminosityBlockCache<C>> {
      using Type = edm::stream::impl::LuminosityBlockCacheHolder<C>;
    };

    template <typename C>
    struct AbilityToImplementor<edm::LuminosityBlockSummaryCache<C>> {
      using Type = edm::stream::impl::LuminosityBlockSummaryCacheHolder<C>;
    };

    template <>
    struct AbilityToImplementor<edm::WatchProcessBlock> {
      using Type = edm::stream::impl::WatchProcessBlock;
    };

    template <>
    struct AbilityToImplementor<edm::BeginProcessBlockProducer> {
      using Type = edm::stream::impl::BeginProcessBlockProducer;
    };

    template <>
    struct AbilityToImplementor<edm::EndProcessBlockProducer> {
      using Type = edm::stream::impl::EndProcessBlockProducer;
    };

    template <>
    struct AbilityToImplementor<edm::BeginRunProducer> {
      using Type = edm::stream::impl::BeginRunProducer;
    };

    template <>
    struct AbilityToImplementor<edm::EndRunProducer> {
      using Type = edm::stream::impl::EndRunProducer;
    };

    template <>
    struct AbilityToImplementor<edm::BeginLuminosityBlockProducer> {
      using Type = edm::stream::impl::BeginLuminosityBlockProducer;
    };

    template <>
    struct AbilityToImplementor<edm::EndLuminosityBlockProducer> {
      using Type = edm::stream::impl::EndLuminosityBlockProducer;
    };

    template <>
    struct AbilityToImplementor<edm::ExternalWork> {
      using Type = edm::stream::impl::ExternalWork;
    };

    // As currently implemented this ability only works
    // with EDProducer, not with EDAnalyzers or EDFilters!
    template <>
    struct AbilityToImplementor<edm::Transformer> {
      using Type = edm::stream::impl::Transformer;
    };

    // As currently implemented this ability only works
    // with EDProducer, not with EDAnalyzers or EDFilters!
    template <>
    struct AbilityToImplementor<edm::Accumulator> {
      using Type = edm::stream::impl::Accumulator;
    };
  }  // namespace stream
}  // namespace edm

#endif
