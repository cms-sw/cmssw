// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TkEmParticle
// 

#include "DataFormats/L1TrackTrigger/interface/L1TkElectronParticle.h"


using namespace l1t ;


L1TkElectronParticle::L1TkElectronParticle()
{
}

L1TkElectronParticle::L1TkElectronParticle( const LorentzVector& p4,
         const edm::Ref< EGammaBxCollection >& egRef,
         const edm::Ptr< L1TTTrackType >& trkPtr,
	 float tkisol )
   : L1TkEmParticle( p4, egRef, tkisol) ,
     trkPtr_ ( trkPtr )

{

 if ( trkPtr_.isNonnull() ) {
	float z = getTrkPtr() -> getPOCA().z();
	setTrkzVtx( z );
 }
}



