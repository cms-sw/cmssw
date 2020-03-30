#ifndef TkTrigger_L1ElectronFwd_h
#define TkTrigger_L1ElectronFwd_h

// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     TkElectronFwd
//

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace l1t {
  class TkElectron;

  typedef std::vector<TkElectron> TkElectronCollection;

  typedef edm::Ref<TkElectronCollection> TkElectronRef;
  typedef edm::RefVector<TkElectronCollection> TkElectronRefVector;
  typedef std::vector<TkElectronRef> TkElectronVectorRef;
}  // namespace l1t
#endif
