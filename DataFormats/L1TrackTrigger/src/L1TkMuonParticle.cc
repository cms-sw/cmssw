// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TkMuonParticle

#include "DataFormats/L1TrackTrigger/interface/L1TkMuonParticle.h"

using namespace l1t;


L1TkMuonParticle::L1TkMuonParticle( const LorentzVector& p4,
         const edm::Ref< RegionalMuonCandBxCollection > &muRef,
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



L1TkMuonParticle::L1TkMuonParticle( const LorentzVector& p4,
         const edm::Ptr< L1TTTrackType >& trkPtr,
         float tkisol )
   : L1Candidate( p4 ),
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



void L1TkMuonParticle::addBarrelStub(const L1MuKBMTCombinedStubRef& stub) {
  barrelStubs_.push_back(stub);
}


