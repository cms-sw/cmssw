// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1CaloTkTauParticle
// 

#include "DataFormats/L1TrackTrigger/interface/L1CaloTkTauParticle.h"


using namespace l1t ;


L1CaloTkTauParticle::L1CaloTkTauParticle()
{
}

L1CaloTkTauParticle::L1CaloTkTauParticle( const LorentzVector& p4,
				const std::vector< L1TTTrackRefPtr >& clustTracks,
    		    Tau& caloTau,
				float vtxIso ,
				float Et)
  : L1Candidate  ( p4 ),
    clustTracks_ ( clustTracks ),
    caloTau_     ( caloTau ),
    vtxIso_      ( vtxIso ), 
    Et_          ( Et)
{

}





