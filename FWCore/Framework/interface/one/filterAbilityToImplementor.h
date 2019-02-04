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
      template<typename T> struct AbilityToImplementor;
      
      template<>
      struct AbilityToImplementor<edm::one::SharedResources> {
        typedef edm::one::impl::SharedResourcesUser<edm::one::EDFilterBase> Type;
      };
      
      template<>
      struct AbilityToImplementor<edm::one::WatchRuns> {
        typedef edm::one::impl::RunWatcher<edm::one::EDFilterBase> Type;
      };

      template<>
      struct AbilityToImplementor<edm::one::WatchLuminosityBlocks> {
        typedef edm::one::impl::LuminosityBlockWatcher<edm::one::EDFilterBase> Type;
      };
      
      template<>
      struct AbilityToImplementor<edm::BeginRunProducer> {
        typedef edm::one::impl::BeginRunProducer<edm::one::EDFilterBase> Type;
      };

      template<>
      struct AbilityToImplementor<edm::EndRunProducer> {
        typedef edm::one::impl::EndRunProducer<edm::one::EDFilterBase> Type;
      };

      template<>
      struct AbilityToImplementor<edm::BeginLuminosityBlockProducer> {
        typedef edm::one::impl::BeginLuminosityBlockProducer<edm::one::EDFilterBase> Type;
      };
      
      template<>
      struct AbilityToImplementor<edm::EndLuminosityBlockProducer> {
        typedef edm::one::impl::EndLuminosityBlockProducer<edm::one::EDFilterBase> Type;
      };
      
      template<typename C>
      struct AbilityToImplementor<edm::RunCache<C>> {
        typedef edm::one::impl::RunCacheHolder<edm::one::EDFilterBase,C> Type;
      };

      template<typename C>
      struct AbilityToImplementor<edm::LuminosityBlockCache<C>> {
        typedef edm::one::impl::LuminosityBlockCacheHolder<edm::one::EDFilterBase,C> Type;
      };

    }
  }
}


#endif
