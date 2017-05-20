// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TkEmParticle
// 

#include "DataFormats/L1TrackTrigger/interface/L1TkTauParticle.h"


using namespace l1t;


L1TkTauParticle::L1TkTauParticle()
{
}

L1TkTauParticle::L1TkTauParticle( const LorentzVector& p4,
	 const edm::Ref< TauBxCollection > &tauCaloRef,
         const edm::Ptr< L1TTTrackType >& trkPtr,
         const edm::Ptr< L1TTTrackType >& trkPtr2,
         const edm::Ptr< L1TTTrackType >& trkPtr3,
	 float tkisol )
   : L1Candidate( p4 ),
     tauCaloRef_ ( tauCaloRef ) ,
     trkPtr_ ( trkPtr ) ,
     trkPtr2_ ( trkPtr2 ) ,
     trkPtr3_ ( trkPtr3 ) ,
     TrkIsol_ ( tkisol )

{

 if ( trkPtr_.isNonnull() ) {
	float z = getTrkPtr() -> getPOCA().z();
	setTrkzVtx( z );
 }
}





