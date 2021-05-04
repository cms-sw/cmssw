// -*- C++ -*-

// Implementation:
//     Explicitly instantiate implementor templates for EDAnalyzerBase
//
// Original Author:  W. David Dagenhart
//         Created:  23 June 2020

#include "FWCore/Framework/src/limited/implementorsMethods.h"
#include "FWCore/Framework/interface/limited/EDAnalyzerBase.h"

namespace edm {
  namespace limited {
    namespace impl {
      template class WatchProcessBlock<edm::limited::EDAnalyzerBase>;
    }  // namespace impl
  }    // namespace limited
}  // namespace edm
