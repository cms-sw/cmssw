// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TkEGTauParticle
// 

#include "DataFormats/L1TrackTrigger/interface/L1TkEGTauParticle.h"


using namespace l1t ;


L1TkEGTauParticle::L1TkEGTauParticle()
{
}

L1TkEGTauParticle::L1TkEGTauParticle( const LorentzVector& p4,
				      const std::vector< L1TTTrackRefPtr >& clustTracks,
				      const std::vector< EGammaRef >& clustEGs,
				      float vtxIso )
  : L1Candidate  ( p4 ),
    clustTracks_ ( clustTracks ),
    clustEGs_ ( clustEGs ),
    vtxIso_      ( vtxIso ) 
{

}





