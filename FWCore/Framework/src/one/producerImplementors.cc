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
// $Id: producerImplementors.cc,v 1.1 2013/05/17 14:49:46 chrjones Exp $
//

// system include files

// user include files
#include "FWCore/Framework/src/one/implementorsMethods.h"
#include "FWCore/Framework/interface/one/EDProducerBase.h"

namespace edm {
  namespace one {
    namespace impl {
      template class RunWatcher<edm::one::EDProducerBase>;
      template class LuminosityBlockWatcher<edm::one::EDProducerBase>;
      template class BeginRunProducer<edm::one::EDProducerBase>;
      template class EndRunProducer<edm::one::EDProducerBase>;
      template class BeginLuminosityBlockProducer<edm::one::EDProducerBase>;
      template class EndLuminosityBlockProducer<edm::one::EDProducerBase>;
    }
  }
}
