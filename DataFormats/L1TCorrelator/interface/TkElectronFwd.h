#ifndef DataFormatsL1TCorrelator_TkElectronFwd_h
#define DataFormatsL1TCorrelator_TkElectronFwd_h

// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     TkElectronFwd
//

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/L1Trigger/interface/RegionalOutput.h"

namespace l1t {
  class TkElectron;

  typedef std::vector<TkElectron> TkElectronCollection;

  typedef edm::Ref<TkElectronCollection> TkElectronRef;
  typedef edm::RefVector<TkElectronCollection> TkElectronRefVector;
  typedef std::vector<TkElectronRef> TkElectronVectorRef;
  typedef l1t::RegionalOutput<l1t::TkElectronCollection> TkElectronRegionalOutput;

}  // namespace l1t
#endif
