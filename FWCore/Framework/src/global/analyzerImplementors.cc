// -*- C++ -*-
//
// Implementation:
//     Explicitly instantiate implementor templates for EDAnalyzerBase
//
// Original Author:  W. David Dagenhart
//         Created:  23 June 2020

#include "FWCore/Framework/src/global/implementorsMethods.h"
#include "FWCore/Framework/interface/global/EDAnalyzerBase.h"

namespace edm {
  namespace global {
    namespace impl {
      template class WatchProcessBlock<edm::global::EDAnalyzerBase>;
    }  // namespace impl
  }    // namespace global
}  // namespace edm
