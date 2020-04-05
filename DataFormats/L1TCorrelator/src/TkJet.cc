// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     TkJet
//

#include "DataFormats/L1TCorrelator/interface/TkJet.h"

using namespace l1t;

TkJet::TkJet() {}

TkJet::TkJet(const LorentzVector& p4,
             const edm::Ref<JetBxCollection>& jetRef,
             const std::vector<edm::Ptr<L1TTTrackType> >& trkPtrs,
             float jetvtx)
    : L1Candidate(p4), jetRef_(jetRef), trkPtrs_(trkPtrs), JetVtx_(jetvtx) {}
TkJet::TkJet(const LorentzVector& p4,
             const std::vector<edm::Ptr<L1TTTrackType> >& trkPtrs,
             float jetvtx,
             unsigned int ntracks,
             unsigned int tighttracks,
             unsigned int displacedtracks,
             unsigned int tightdisplacedtracks)
    : L1Candidate(p4),
      trkPtrs_(trkPtrs),
      JetVtx_(jetvtx),
      ntracks_(ntracks),
      tighttracks_(tighttracks),
      displacedtracks_(displacedtracks),
      tightdisplacedtracks_(tightdisplacedtracks) {}
int TkJet::bx() const {
  // in the producer TkJetProducer.cc, we keep only jets with bx = 0
  return 0;
}
