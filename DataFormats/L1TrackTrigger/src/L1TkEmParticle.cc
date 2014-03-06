// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TkEmParticle
// 

#include "DataFormats/L1TrackTrigger/interface/L1TkEmParticle.h"


using namespace l1extra ;


L1TkEmParticle::L1TkEmParticle()
{
}

L1TkEmParticle::L1TkEmParticle( const LorentzVector& p4,
         const edm::Ref< L1EmParticleCollection >& egRef,
	 float tkisol )
   : LeafCandidate( ( char ) 0, p4 ),
     egRef_ ( egRef ),
     TrkIsol_ ( tkisol ) 
{

}


int L1TkEmParticle::bx() const {
 int dummy = -999;
 if ( egRef_.isNonnull() ) {
	return (getEGRef() -> bx()) ;
 }
 else {
	return dummy;

 }
}







