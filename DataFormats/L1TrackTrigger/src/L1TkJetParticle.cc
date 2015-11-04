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
				  const std::vector< edm::Ptr< L1TkTrackType > >& trkPtrs,
				  float jetvtx )
: LeafCandidate( ( char ) 0, p4 ),
  jetRef_ ( jetRef ),
  trkPtrs_ ( trkPtrs ),
  JetVtx_ ( jetvtx )
{
}


int L1TkJetParticle::bx() const {

  // in the producer L1TkJetProducer.cc, we keep only jets with bx = 0 
  int dummy = 0;
  return dummy;
  
  /*
    int dummy = -999;
    if ( jetRef_.isNonnull() ) {
    return (getJetRef() -> bx()) ;
    }
    else {
    return dummy;
    
    }
  */
}

