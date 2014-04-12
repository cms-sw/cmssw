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
    template<typename T> struct AbilityToImplementor;
    
    template<typename C>
    struct AbilityToImplementor<edm::GlobalCache<C>> {
      typedef edm::stream::impl::GlobalCacheHolder<C> Type;
    };
    
    template<typename C>
    struct AbilityToImplementor<edm::RunCache<C>> {
      typedef edm::stream::impl::RunCacheHolder<C> Type;
    };
    
    template<typename C>
    struct AbilityToImplementor<edm::RunSummaryCache<C>> {
      typedef edm::stream::impl::RunSummaryCacheHolder<C> Type;
    };
    
    template<typename C>
    struct AbilityToImplementor<edm::LuminosityBlockCache<C>> {
      typedef edm::stream::impl::LuminosityBlockCacheHolder<C> Type;
    };
    
    template<typename C>
    struct AbilityToImplementor<edm::LuminosityBlockSummaryCache<C>> {
      typedef edm::stream::impl::LuminosityBlockSummaryCacheHolder<C> Type;
    };
    
    template<>
    struct AbilityToImplementor<edm::BeginRunProducer> {
      typedef edm::stream::impl::BeginRunProducer Type;
    };
    
    template<>
    struct AbilityToImplementor<edm::EndRunProducer> {
      typedef edm::stream::impl::EndRunProducer Type;
    };
    
    template<>
    struct AbilityToImplementor<edm::BeginLuminosityBlockProducer> {
      typedef edm::stream::impl::BeginLuminosityBlockProducer Type;
    };
    
    template<>
    struct AbilityToImplementor<edm::EndLuminosityBlockProducer> {
      typedef edm::stream::impl::EndLuminosityBlockProducer Type;
    };
  }
}

#endif
