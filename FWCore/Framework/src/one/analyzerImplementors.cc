// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     analyzerImplementors
// 
// Implementation:
//     Explicitly instantiate implementor templates for EDAnalyzerBase
//
// Original Author:  Chris Jones
//         Created:  Thu, 01 Aug 2013 20:14:06 GMT
//

// system include files

// user include files
#include "FWCore/Framework/src/one/implementorsMethods.h"
#include "FWCore/Framework/interface/one/EDAnalyzerBase.h"

namespace edm {
  namespace one {
    namespace impl {
      template class RunWatcher<edm::one::EDAnalyzerBase>;
      template class LuminosityBlockWatcher<edm::one::EDAnalyzerBase>;
    }
  }
}
