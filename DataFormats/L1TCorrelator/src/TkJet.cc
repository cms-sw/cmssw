// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     TkJet
//

#include "DataFormats/L1TCorrelator/interface/TkJet.h"

using namespace l1t;

TkJet::TkJet()
    : JetVtx_(-999.),
      ntracks_(0),
      tighttracks_(0),
      displacedtracks_(0),
      tightdisplacedtracks_(0),
      displacedTag_(false) {}

TkJet::TkJet(const LorentzVector& p4,
             const edm::Ref<JetBxCollection>& jetRef,
             const std::vector<edm::Ptr<L1TTTrackType> >& trkPtrs,
             float jetvtx)
    : L1Candidate(p4),
      jetRef_(jetRef),
      trkPtrs_(trkPtrs),
      JetVtx_(jetvtx),
      ntracks_(0),
      tighttracks_(0),
      displacedtracks_(0),
      tightdisplacedtracks_(0),
      displacedTag_(false) {}
TkJet::TkJet(const LorentzVector& p4,
             const std::vector<edm::Ptr<L1TTTrackType> >& trkPtrs,
             float jetvtx,
             unsigned int ntracks,
             unsigned int tighttracks,
             unsigned int displacedtracks,
             unsigned int tightdisplacedtracks,
             bool displacedTag)
    : L1Candidate(p4),
      trkPtrs_(trkPtrs),
      JetVtx_(jetvtx),
      ntracks_(ntracks),
      tighttracks_(tighttracks),
      displacedtracks_(displacedtracks),
      tightdisplacedtracks_(tightdisplacedtracks),
      displacedTag_(displacedTag) {}

int TkJet::bx() const {
  // in the producer TkJetProducer.cc, we keep only jets with bx = 0
  return 0;
}
