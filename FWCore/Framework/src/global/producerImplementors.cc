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
// $Id$
//

// system include files

// user include files
#include "FWCore/Framework/src/global/implementorsMethods.h"
#include "FWCore/Framework/interface/global/EDProducerBase.h"

namespace edm {
  namespace global {
    namespace impl {
      template class BeginRunProducer<edm::global::EDProducerBase>;
      template class EndRunProducer<edm::global::EDProducerBase>;
      template class BeginLuminosityBlockProducer<edm::global::EDProducerBase>;
      template class EndLuminosityBlockProducer<edm::global::EDProducerBase>;
    }
  }
}
