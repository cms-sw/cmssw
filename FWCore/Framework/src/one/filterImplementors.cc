// -*- C++ -*-
//
// Package:     Package
// Class  :     producerImplementors
//
// Implementation:
//     Explicitly instantiate implementor templates for EDFilterBase
//
// Original Author:  Chris Jones
//         Created:  Thu, 09 May 2013 20:14:06 GMT
//

// system include files

// user include files
#include "FWCore/Framework/src/one/implementorsMethods.h"
#include "FWCore/Framework/interface/one/EDFilterBase.h"

namespace edm {
  namespace one {
    namespace impl {
      template class SharedResourcesUser<edm::one::EDFilterBase>;
      template class RunWatcher<edm::one::EDFilterBase>;
      template class LuminosityBlockWatcher<edm::one::EDFilterBase>;
      template class WatchProcessBlock<edm::one::EDFilterBase>;
      template class BeginProcessBlockProducer<edm::one::EDFilterBase>;
      template class EndProcessBlockProducer<edm::one::EDFilterBase>;
      template class BeginRunProducer<edm::one::EDFilterBase>;
      template class EndRunProducer<edm::one::EDFilterBase>;
      template class BeginLuminosityBlockProducer<edm::one::EDFilterBase>;
      template class EndLuminosityBlockProducer<edm::one::EDFilterBase>;
    }  // namespace impl
  }    // namespace one
}  // namespace edm
