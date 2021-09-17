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
#include "FWCore/Framework/src/global/implementorsMethods.h"
#include "FWCore/Framework/interface/global/EDProducerBase.h"

namespace edm {
  namespace global {
    namespace impl {
      template class WatchProcessBlock<edm::global::EDProducerBase>;
      template class BeginProcessBlockProducer<edm::global::EDProducerBase>;
      template class EndProcessBlockProducer<edm::global::EDProducerBase>;
      template class BeginRunProducer<edm::global::EDProducerBase>;
      template class EndRunProducer<edm::global::EDProducerBase>;
      template class BeginLuminosityBlockProducer<edm::global::EDProducerBase>;
      template class EndLuminosityBlockProducer<edm::global::EDProducerBase>;
      template class ExternalWork<edm::global::EDProducerBase>;
    }  // namespace impl
  }    // namespace global
}  // namespace edm
