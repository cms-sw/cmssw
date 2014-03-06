// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TkEmParticle
// 

#include "DataFormats/L1TrackTrigger/interface/L1TkTauParticle.h"


using namespace l1extra ;


L1TkTauParticle::L1TkTauParticle()
{
}

L1TkTauParticle::L1TkTauParticle( const LorentzVector& p4,
	 const edm::Ref< L1JetParticleCollection > &tauCaloRef,
         const edm::Ptr< L1TkTrackType >& trkPtr,
         const edm::Ptr< L1TkTrackType >& trkPtr2,
         const edm::Ptr< L1TkTrackType >& trkPtr3,
	 float tkisol )
   : LeafCandidate( ( char ) 0, p4 ),
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

int L1TkTauParticle::bx() const {
 int dummy = 0;
 return dummy;
}




