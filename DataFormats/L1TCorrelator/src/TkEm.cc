// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     TkEm
//

#include "DataFormats/L1TCorrelator/interface/TkEm.h"
// FIXME: can remove
#include "DataFormats/Common/interface/RefToPtr.h"

using namespace l1t;

TkEm::TkEm() {}

TkEm::TkEm(const LorentzVector& p4, float tkisol, float tkisolPV)
    : TkEm(p4, edm::Ptr<L1Candidate>(), tkisol, tkisolPV) {}

TkEm::TkEm(const LorentzVector& p4, const edm::Ptr<L1Candidate>& egCaloPtr, float tkisol, float tkisolPV)
    : L1Candidate(p4),
      egCaloPtr_(egCaloPtr),
      trkIsol_(tkisol),
      trkIsolPV_(tkisolPV),
      pfIsol_(-999),
      pfIsolPV_(-999),
      puppiIsol_(-999),
      puppiIsolPV_(-999),
      egBinaryWord0_(0),
      egBinaryWord1_(0),
      egBinaryWord2_(0),
      encoding_(HWEncoding::None) {}
