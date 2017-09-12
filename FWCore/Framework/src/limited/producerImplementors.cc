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
#include "FWCore/Framework/src/limited/implementorsMethods.h"
#include "FWCore/Framework/interface/limited/EDProducerBase.h"

namespace edm {
  namespace limited {
    namespace impl {
      template class BeginRunProducer<edm::limited::EDProducerBase>;
      template class EndRunProducer<edm::limited::EDProducerBase>;
      template class BeginLuminosityBlockProducer<edm::limited::EDProducerBase>;
      template class EndLuminosityBlockProducer<edm::limited::EDProducerBase>;
    }
  }
}
