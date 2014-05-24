// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TkMuonParticle

#include "DataFormats/L1TrackTrigger/interface/L1TkMuonParticle.h"

using namespace l1extra;

L1TkMuonParticle::L1TkMuonParticle() {}

L1TkMuonParticle::L1TkMuonParticle( const LorentzVector& p4,
                                    const edm::Ptr< DTMatch > &muRef,
                                    float tkisol )
 : LeafCandidate( ( char ) 0, p4 ),
   theDTMatch ( muRef ) ,
   theIsolation ( tkisol )
{
  // other constructor operations if needed

}

