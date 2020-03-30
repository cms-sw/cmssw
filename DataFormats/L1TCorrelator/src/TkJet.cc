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
             //				  const edm::Ref< JetBxCollection >& jetRef,
             const std::vector<edm::Ptr<L1TTTrackType> >& trkPtrs,
             float jetvtx,
             unsigned int ntracks,
             unsigned int tighttracks,
             unsigned int displacedtracks,
             unsigned int tightdisplacedtracks
             // TkJetDisp counters
             )
    : L1Candidate(p4),
      //jetRef_ ( jetRef ),
      trkPtrs_(trkPtrs),
      JetVtx_(jetvtx),
      ntracks_(ntracks),
      tighttracks_(tighttracks),
      displacedtracks_(displacedtracks),
      tightdisplacedtracks_(tightdisplacedtracks) {
  //TkJetDisp DispCounters(ntracks,tighttracks, displacedtracks, tightdisplacedtracks);
  //setDispCounters(DispCounters);
}
int TkJet::bx() const {
  // in the producer TkJetProducer.cc, we keep only jets with bx = 0
  int dummy = 0;
  return dummy;

  /*
    int dummy = -999;
    if ( jetRef_.isNonnull() ) {
    return (jetRef() -> bx()) ;
    }
    else {
    return dummy;
    
    }
  */
}
