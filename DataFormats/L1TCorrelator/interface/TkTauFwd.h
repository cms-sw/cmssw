#ifndef DataFormatsL1TCorrelator_TkTauFwd_h
#define DataFormatsL1TCorrelator_TkTauFwd_h

// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     TkTauFwd
//

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace l1t {

  class TkTau;

  typedef std::vector<TkTau> TkTauCollection;

  typedef edm::Ref<TkTauCollection> TkTauRef;
  typedef edm::RefVector<TkTauCollection> TkTauRefVector;
  typedef std::vector<TkTauRef> TkTauVectorRef;
}  // namespace l1t

#endif
