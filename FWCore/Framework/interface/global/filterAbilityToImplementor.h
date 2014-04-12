#ifndef FWCore_Framework_global_filterAbilityToImplementor_h
#define FWCore_Framework_global_filterAbilityToImplementor_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// File  :     filterAbilityToImplementor
// 
/**\file  filterAbilityToImplementor.h "FWCore/Framework/interface/global/filterAbilityToImplementor.h"

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
#include "FWCore/Framework/interface/global/EDFilterBase.h"

// forward declarations
namespace edm {
  namespace global {
    namespace filter {
      template<typename T> struct AbilityToImplementor;
      
      template<typename C>
      struct AbilityToImplementor<edm::StreamCache<C>> {
        typedef edm::global::impl::StreamCacheHolder<edm::global::EDFilterBase,C> Type;
      };

      template<typename C>
      struct AbilityToImplementor<edm::RunCache<C>> {
        typedef edm::global::impl::RunCacheHolder<edm::global::EDFilterBase,C> Type;
      };
      
      template<typename C>
      struct AbilityToImplementor<edm::RunSummaryCache<C>> {
        typedef edm::global::impl::RunSummaryCacheHolder<edm::global::EDFilterBase,C> Type;
      };
      
      template<typename C>
      struct AbilityToImplementor<edm::LuminosityBlockCache<C>> {
        typedef edm::global::impl::LuminosityBlockCacheHolder<edm::global::EDFilterBase,C> Type;
      };
      
      template<typename C>
      struct AbilityToImplementor<edm::LuminosityBlockSummaryCache<C>> {
        typedef edm::global::impl::LuminosityBlockSummaryCacheHolder<edm::global::EDFilterBase,C> Type;
      };
      
      template<>
      struct AbilityToImplementor<edm::BeginRunProducer> {
        typedef edm::global::impl::BeginRunProducer<edm::global::EDFilterBase> Type;
      };
      
      template<>
      struct AbilityToImplementor<edm::EndRunProducer> {
        typedef edm::global::impl::EndRunProducer<edm::global::EDFilterBase> Type;
      };
      
      template<>
      struct AbilityToImplementor<edm::BeginLuminosityBlockProducer> {
        typedef edm::global::impl::BeginLuminosityBlockProducer<edm::global::EDFilterBase> Type;
      };
      
      template<>
      struct AbilityToImplementor<edm::EndLuminosityBlockProducer> {
        typedef edm::global::impl::EndLuminosityBlockProducer<edm::global::EDFilterBase> Type;
      };
      
      template<bool,bool,typename T> struct SpecializeAbilityToImplementor {
        typedef typename AbilityToImplementor<T>::Type Type;
      };
      
      template<bool B,typename C> struct SpecializeAbilityToImplementor<true,B,edm::RunSummaryCache<C>> {
        typedef typename edm::global::impl::EndRunSummaryProducer<edm::global::EDFilterBase,C> Type;
      };
      
      template<bool B> struct SpecializeAbilityToImplementor<true,B,edm::EndRunProducer> {
        typedef typename edm::global::impl::EmptyType Type;
      };

      template<bool B,typename C> struct SpecializeAbilityToImplementor<B,true,edm::LuminosityBlockSummaryCache<C>> {
        typedef typename edm::global::impl::EndLuminosityBlockSummaryProducer<edm::global::EDFilterBase,C> Type;
      };
      
      template<bool B> struct SpecializeAbilityToImplementor<B,true,edm::EndLuminosityBlockProducer> {
        typedef typename edm::global::impl::EmptyType Type;
      };
    }
  }
}

#endif
