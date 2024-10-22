// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1CaloTkTau
//

#include "DataFormats/L1TCorrelator/interface/L1CaloTkTau.h"

using namespace l1t;

L1CaloTkTau::L1CaloTkTau() {}

L1CaloTkTau::L1CaloTkTau(const LorentzVector& p4,
                         const LorentzVector& tracksP4,
                         const std::vector<L1TTTrackRefPtr>& clustTracks,
                         Tau& caloTau,
                         float vtxIso)
    : L1Candidate(p4), tracksP4_(tracksP4), clustTracks_(clustTracks), caloTau_(caloTau), vtxIso_(vtxIso) {}
