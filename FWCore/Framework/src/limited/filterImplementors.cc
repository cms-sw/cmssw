// -*- C++ -*-
//
// Package:     Package
// Class  :     filterImplementors
// 
// Implementation:
//     Explicitly instantiate implementor templates for EDFilterBase
//
// Original Author:  Chris Jones
//         Created:  Thu, 09 May 2013 20:14:06 GMT
//

// system include files

// user include files
#include "FWCore/Framework/src/limited/implementorsMethods.h"
#include "FWCore/Framework/interface/limited/EDFilterBase.h"

namespace edm {
  namespace limited {
    namespace impl {
      template class BeginRunProducer<edm::limited::EDFilterBase>;
      template class EndRunProducer<edm::limited::EDFilterBase>;
      template class BeginLuminosityBlockProducer<edm::limited::EDFilterBase>;
      template class EndLuminosityBlockProducer<edm::limited::EDFilterBase>;
    }
  }
}
