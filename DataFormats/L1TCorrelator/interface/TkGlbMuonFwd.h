#ifndef DataFormatsL1TCorrelator_TkGlbMuonFwd_h
#define DataFormatsL1TCorrelator_TkGlbMuonFwd_h

// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     TkGlbMuonFwd
//

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace l1t {

  class TkGlbMuon;

  typedef std::vector<TkGlbMuon> TkGlbMuonCollection;

  typedef edm::Ref<TkGlbMuonCollection> TkGlbMuonRef;
  typedef edm::RefVector<TkGlbMuonCollection> TkGlbMuonRefVector;
  typedef std::vector<TkGlbMuonRef> TkGlbMuonVectorRef;
}  // namespace l1t

#endif
