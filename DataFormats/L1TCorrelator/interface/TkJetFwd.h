#ifndef DataFormatsL1TCorrelator_TkJetFwd_h
#define DataFormatsL1TCorrelator_TkJetFwd_h

// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     TkJetFwd
//

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace l1t {
  namespace io_v1 {
    class TkJet;
  }
  using TkJet = io_v1::TkJet;

  typedef edm::RefProd<TkJet> TkJetRefProd;

  typedef std::vector<TkJet> TkJetCollection;

  typedef edm::Ref<TkJetCollection> TkJetRef;
  typedef edm::RefVector<TkJetCollection> TkJetRefVector;
  typedef std::vector<TkJetRef> TkJetVectorRef;
}  // namespace l1t

#endif
