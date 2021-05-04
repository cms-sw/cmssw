// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     TkEm
//

#include "DataFormats/L1TCorrelator/interface/TkEm.h"

using namespace l1t;

TkEm::TkEm() {}

TkEm::TkEm(const LorentzVector& p4, const edm::Ref<EGammaBxCollection>& egRef, float tkisol)
    : L1Candidate(p4), egRef_(egRef), trkIsol_(tkisol), trkIsolPV_(-999) {}

TkEm::TkEm(const LorentzVector& p4, const edm::Ref<EGammaBxCollection>& egRef, float tkisol, float tkisolPV)
    : L1Candidate(p4), egRef_(egRef), trkIsol_(tkisol), trkIsolPV_(tkisolPV) {}
