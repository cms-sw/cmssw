// -*- C++ -*-
//
// Package:     Package
// Class  :     producerImplementors
//
// Implementation:
//     Explicitly instantiate implementor templates for EDProducerBase
//
// Original Author:  Chris Jones
//         Created:  Thu, 09 May 2013 20:14:06 GMT
//

// system include files

// user include files
#include "FWCore/Framework/src/one/implementorsMethods.h"
#include "FWCore/Framework/interface/one/EDProducerBase.h"

namespace edm {
  namespace one {
    namespace impl {
      template class SharedResourcesUser<edm::one::EDProducerBase>;
      template class RunWatcher<edm::one::EDProducerBase>;
      template class LuminosityBlockWatcher<edm::one::EDProducerBase>;
      template class WatchProcessBlock<edm::one::EDProducerBase>;
      template class BeginProcessBlockProducer<edm::one::EDProducerBase>;
      template class EndProcessBlockProducer<edm::one::EDProducerBase>;
      template class BeginRunProducer<edm::one::EDProducerBase>;
      template class EndRunProducer<edm::one::EDProducerBase>;
      template class BeginLuminosityBlockProducer<edm::one::EDProducerBase>;
      template class EndLuminosityBlockProducer<edm::one::EDProducerBase>;
    }  // namespace impl
  }    // namespace one
}  // namespace edm
