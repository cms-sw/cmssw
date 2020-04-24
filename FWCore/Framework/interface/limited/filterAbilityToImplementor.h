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
      template<typename T> struct AbilityToImplementor;
      
      template<typename C>
      struct AbilityToImplementor<edm::StreamCache<C>> {
        typedef edm::limited::impl::StreamCacheHolder<edm::limited::EDFilterBase,C> Type;
      };

      template<typename C>
      struct AbilityToImplementor<edm::RunCache<C>> {
        typedef edm::limited::impl::RunCacheHolder<edm::limited::EDFilterBase,C> Type;
      };
      
      template<typename C>
      struct AbilityToImplementor<edm::RunSummaryCache<C>> {
        typedef edm::limited::impl::RunSummaryCacheHolder<edm::limited::EDFilterBase,C> Type;
      };
      
      template<typename C>
      struct AbilityToImplementor<edm::LuminosityBlockCache<C>> {
        typedef edm::limited::impl::LuminosityBlockCacheHolder<edm::limited::EDFilterBase,C> Type;
      };
      
      template<typename C>
      struct AbilityToImplementor<edm::LuminosityBlockSummaryCache<C>> {
        typedef edm::limited::impl::LuminosityBlockSummaryCacheHolder<edm::limited::EDFilterBase,C> Type;
      };
      
      template<>
      struct AbilityToImplementor<edm::BeginRunProducer> {
        typedef edm::limited::impl::BeginRunProducer<edm::limited::EDFilterBase> Type;
      };
      
      template<>
      struct AbilityToImplementor<edm::EndRunProducer> {
        typedef edm::limited::impl::EndRunProducer<edm::limited::EDFilterBase> Type;
      };
      
      template<>
      struct AbilityToImplementor<edm::BeginLuminosityBlockProducer> {
        typedef edm::limited::impl::BeginLuminosityBlockProducer<edm::limited::EDFilterBase> Type;
      };
      
      template<>
      struct AbilityToImplementor<edm::EndLuminosityBlockProducer> {
        typedef edm::limited::impl::EndLuminosityBlockProducer<edm::limited::EDFilterBase> Type;
      };
      
      template<bool,bool,typename T> struct SpecializeAbilityToImplementor {
        typedef typename AbilityToImplementor<T>::Type Type;
      };
      
      template<bool B,typename C> struct SpecializeAbilityToImplementor<true,B,edm::RunSummaryCache<C>> {
        typedef typename edm::limited::impl::EndRunSummaryProducer<edm::limited::EDFilterBase,C> Type;
      };
      
      template<bool B> struct SpecializeAbilityToImplementor<true,B,edm::EndRunProducer> {
        typedef typename edm::limited::impl::EmptyType Type;
      };

      template<bool B,typename C> struct SpecializeAbilityToImplementor<B,true,edm::LuminosityBlockSummaryCache<C>> {
        typedef typename edm::limited::impl::EndLuminosityBlockSummaryProducer<edm::limited::EDFilterBase,C> Type;
      };
      
      template<bool B> struct SpecializeAbilityToImplementor<B,true,edm::EndLuminosityBlockProducer> {
        typedef typename edm::limited::impl::EmptyType Type;
      };
    }
  }
}

#endif
