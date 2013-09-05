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
      template<typename T> struct AbilityToImplementor;
      
      template<typename C>
      struct AbilityToImplementor<edm::StreamCache<C>> {
        typedef edm::global::impl::StreamCacheHolder<edm::global::EDAnalyzerBase,C> Type;
      };

      template<typename C>
      struct AbilityToImplementor<edm::RunCache<C>> {
        typedef edm::global::impl::RunCacheHolder<edm::global::EDAnalyzerBase,C> Type;
      };
      
      template<typename C>
      struct AbilityToImplementor<edm::RunSummaryCache<C>> {
        typedef edm::global::impl::RunSummaryCacheHolder<edm::global::EDAnalyzerBase,C> Type;
      };
      
      template<typename C>
      struct AbilityToImplementor<edm::LuminosityBlockCache<C>> {
        typedef edm::global::impl::LuminosityBlockCacheHolder<edm::global::EDAnalyzerBase,C> Type;
      };
      
      template<typename C>
      struct AbilityToImplementor<edm::LuminosityBlockSummaryCache<C>> {
        typedef edm::global::impl::LuminosityBlockSummaryCacheHolder<edm::global::EDAnalyzerBase,C> Type;
      };
    }
  }
}

#endif
