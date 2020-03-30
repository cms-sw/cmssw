// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TrkTau
//

#include "DataFormats/L1TCorrelator/interface/L1TrkTau.h"

using namespace l1t;

L1TrkTau::L1TrkTau() {}

L1TrkTau::L1TrkTau(const LorentzVector& p4, const std::vector<L1TTTrackRefPtr>& clustTracks, float iso)
    : L1Candidate(p4), clustTracks_(clustTracks), iso_(iso) {}
