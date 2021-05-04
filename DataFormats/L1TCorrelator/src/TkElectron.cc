// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     TkEm
//

#include "DataFormats/L1TCorrelator/interface/TkElectron.h"

using namespace l1t;

TkElectron::TkElectron() {}

TkElectron::TkElectron(const LorentzVector& p4,
                       const edm::Ref<EGammaBxCollection>& egRef,
                       const edm::Ptr<L1TTTrackType>& trackPtr,
                       float tkisol)
    : TkEm(p4, egRef, tkisol, -999),
      trkPtr_(trackPtr)

{
  if (trkPtr_.isNonnull()) {
    setTrkzVtx(trkPtr()->POCA().z());
  }
}
