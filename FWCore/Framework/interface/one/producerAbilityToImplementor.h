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
      template<typename T> struct AbilityToImplementor;
      
      template<>
      struct AbilityToImplementor<edm::one::SharedResources> {
        typedef edm::one::impl::SharedResourcesUser Type;
      };
      
      template<>
      struct AbilityToImplementor<edm::one::WatchRuns> {
        typedef edm::one::impl::RunWatcher<edm::one::EDProducerBase> Type;
      };

      template<>
      struct AbilityToImplementor<edm::one::WatchLuminosityBlocks> {
        typedef edm::one::impl::LuminosityBlockWatcher<edm::one::EDProducerBase> Type;
      };
      
      template<>
      struct AbilityToImplementor<edm::BeginRunProducer> {
        typedef edm::one::impl::BeginRunProducer<edm::one::EDProducerBase> Type;
      };

      template<>
      struct AbilityToImplementor<edm::EndRunProducer> {
        typedef edm::one::impl::EndRunProducer<edm::one::EDProducerBase> Type;
      };

      template<>
      struct AbilityToImplementor<edm::BeginLuminosityBlockProducer> {
        typedef edm::one::impl::BeginLuminosityBlockProducer<edm::one::EDProducerBase> Type;
      };
      
      template<>
      struct AbilityToImplementor<edm::EndLuminosityBlockProducer> {
        typedef edm::one::impl::EndLuminosityBlockProducer<edm::one::EDProducerBase> Type;
      };
    }
  }
}


#endif
