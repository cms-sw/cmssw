// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TrkTauParticle
// 

#include "DataFormats/L1TrackTrigger/interface/L1TrkTauParticle.h"


using namespace l1t ;


L1TrkTauParticle::L1TrkTauParticle()
{
}

L1TrkTauParticle::L1TrkTauParticle( const LorentzVector& p4,
				    const std::vector< L1TTTrackRefPtr >& clustTracks,
				    float iso )
  : L1Candidate  ( p4 ),
    clustTracks_ ( clustTracks ),
    iso_      ( iso ) 
{

}





