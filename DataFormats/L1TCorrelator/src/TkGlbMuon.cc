// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     TkGlbMuon

#include "DataFormats/L1TCorrelator/interface/TkGlbMuon.h"

using namespace l1t;

TkGlbMuon::TkGlbMuon(const LorentzVector& p4,
                     const edm::Ref<MuonBxCollection>& muRef,
                     const edm::Ptr<L1TTTrackType>& trackPtr,
                     float tkisol)
    : L1Candidate(p4), muRef_(muRef), trkPtr_(trackPtr), theIsolation(tkisol), TrkzVtx_(999), quality_(999) {
  if (trkPtr_.isNonnull()) {
    setTrkzVtx(trkPtr()->POCA().z());
  }
}
