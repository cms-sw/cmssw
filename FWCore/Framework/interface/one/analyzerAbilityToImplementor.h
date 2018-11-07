#ifndef FWCore_Framework_one_analyzerAbilityToImplementor_h
#define FWCore_Framework_one_analyzerAbilityToImplementor_h
// -*- C++ -*-
//
// Package:     Package
// Class  :     analyzer::AbilityToImplementor
// 
/**\class analyzer::AbilityToImplementor analyzerAbilityToImplementor.h "FWCore/Framework/interface/one/analyzerAbilityToImplementor.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu, 01 Aug 2013 19:39:58 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/moduleAbilities.h"
#include "FWCore/Framework/interface/one/moduleAbilities.h"
#include "FWCore/Framework/interface/one/implementors.h"
#include "FWCore/Framework/interface/one/EDAnalyzerBase.h"

// forward declarations

namespace edm {
  namespace one {
    namespace analyzer {
      template<typename T> struct AbilityToImplementor;
      
      template<>
      struct AbilityToImplementor<edm::one::SharedResources> {
        typedef edm::one::impl::SharedResourcesUser<edm::one::EDAnalyzerBase> Type;
      };
      
      template<>
      struct AbilityToImplementor<edm::one::WatchRuns> {
        typedef edm::one::impl::RunWatcher<edm::one::EDAnalyzerBase> Type;
      };

      template<>
      struct AbilityToImplementor<edm::one::WatchLuminosityBlocks> {
        typedef edm::one::impl::LuminosityBlockWatcher<edm::one::EDAnalyzerBase> Type;
      };
      
      template<typename C>
      struct AbilityToImplementor<edm::RunCache<C>> {
        typedef edm::one::impl::RunCacheHolder<edm::one::EDAnalyzerBase,C> Type;
      };

      template<typename C>
      struct AbilityToImplementor<edm::LuminosityBlockCache<C>> {
        typedef edm::one::impl::LuminosityBlockCacheHolder<edm::one::EDAnalyzerBase,C> Type;
      };
    }
  }
}


#endif
