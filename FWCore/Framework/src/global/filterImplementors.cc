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
#include "FWCore/Framework/src/global/implementorsMethods.h"
#include "FWCore/Framework/interface/global/EDFilterBase.h"

namespace edm {
  namespace global {
    namespace impl {
      template class WatchProcessBlock<edm::global::EDFilterBase>;
      template class BeginProcessBlockProducer<edm::global::EDFilterBase>;
      template class EndProcessBlockProducer<edm::global::EDFilterBase>;
      template class BeginRunProducer<edm::global::EDFilterBase>;
      template class EndRunProducer<edm::global::EDFilterBase>;
      template class BeginLuminosityBlockProducer<edm::global::EDFilterBase>;
      template class EndLuminosityBlockProducer<edm::global::EDFilterBase>;
      template class ExternalWork<edm::global::EDFilterBase>;
    }  // namespace impl
  }    // namespace global
}  // namespace edm
