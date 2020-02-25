// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TkGlbMuonParticle

#include "DataFormats/L1TrackTrigger/interface/L1TkGlbMuonParticle.h"

using namespace l1t;


L1TkGlbMuonParticle::L1TkGlbMuonParticle( const LorentzVector& p4,
         const edm::Ref< MuonBxCollection > &muRef,
         const edm::Ptr< L1TTTrackType >& trkPtr,
         float tkisol )
   : L1Candidate( p4 ),
     muRef_ ( muRef ) ,	
     trkPtr_ ( trkPtr ) ,
     theIsolation ( tkisol ),
     TrkzVtx_(999),
     quality_(999)
{

 if ( trkPtr_.isNonnull() ) {
	float z = getTrkPtr() -> getPOCA().z();
	setTrkzVtx( z );
 }
}

