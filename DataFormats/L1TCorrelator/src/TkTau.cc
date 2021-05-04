// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     TkEm
//

#include "DataFormats/L1TCorrelator/interface/TkTau.h"

using namespace l1t;

TkTau::TkTau() {}

TkTau::TkTau(const LorentzVector& p4,
             const edm::Ref<TauBxCollection>& tauCaloRef,
             const edm::Ptr<L1TTTrackType>& trackPtr,
             const edm::Ptr<L1TTTrackType>& trackPtr2,
             const edm::Ptr<L1TTTrackType>& trackPtr3,
             float tkisol)
    : L1Candidate(p4),
      tauCaloRef_(tauCaloRef),
      trkPtr_(trackPtr),
      trkPtr2_(trackPtr2),
      trkPtr3_(trackPtr3),
      TrkIsol_(tkisol)

{
  if (trkPtr_.isNonnull()) {
    setTrkzVtx(trkPtr()->POCA().z());
  }
}
