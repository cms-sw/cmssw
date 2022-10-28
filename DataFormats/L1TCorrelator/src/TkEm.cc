// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     TkEm
//

#include "DataFormats/L1TCorrelator/interface/TkEm.h"

using namespace l1t;

TkEm::TkEm() {}

TkEm::TkEm(const LorentzVector& p4, const edm::Ref<EGammaBxCollection>& egRef, float tkisol)
    : TkEm(p4, egRef, tkisol, -999) {}

TkEm::TkEm(const LorentzVector& p4, const edm::Ref<EGammaBxCollection>& egRef, float tkisol, float tkisolPV)
    : L1Candidate(p4),
      egRef_(egRef),
      trkIsol_(tkisol),
      trkIsolPV_(tkisolPV),
      pfIsol_(-999),
      pfIsolPV_(-999),
      puppiIsol_(-999),
      puppiIsolPV_(-999),
      egBinaryWord0_(0),
      egBinaryWord1_(0),
      egBinaryWord2_(0) {}
