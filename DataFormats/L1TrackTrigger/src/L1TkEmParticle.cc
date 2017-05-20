// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TkEmParticle
// 

#include "DataFormats/L1TrackTrigger/interface/L1TkEmParticle.h"


using namespace l1t ;


L1TkEmParticle::L1TkEmParticle()
{
}

L1TkEmParticle::L1TkEmParticle( const LorentzVector& p4,
         const edm::Ref< EGammaBxCollection >& egRef,
	 float tkisol )
   : L1Candidate( p4 ),
     egRef_ ( egRef ),
     TrkIsol_ ( tkisol ) 
{

}






