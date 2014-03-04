// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TkJetParticle
// 

#include "DataFormats/L1TrackTrigger/interface/L1TkJetParticle.h"


using namespace l1extra ;


L1TkJetParticle::L1TkJetParticle()
{
}

L1TkJetParticle::L1TkJetParticle( const LorentzVector& p4,
         const edm::Ref< L1JetParticleCollection >& jetRef,
         float jetvtx )
   : LeafCandidate( ( char ) 0, p4 ),
     jetRef_ ( jetRef ),
     JetVtx_ ( jetvtx )
{

}


int L1TkJetParticle::bx() const {
 int dummy = -999;
 if ( jetRef_.isNonnull() ) {
        return (getJetRef() -> bx()) ;
 }
 else {
        return dummy;

 }
}

