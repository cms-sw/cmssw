// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TkMuonParticle

#include "DataFormats/L1TrackTrigger/interface/L1TkMuonParticle.h"

using namespace l1extra;

L1TkMuonParticle::L1TkMuonParticle() {}

	// Padova's TkMuons
L1TkMuonParticle::L1TkMuonParticle( const LorentzVector& p4,
                                    const edm::Ptr< DTMatch > &muRef,
                                    float tkisol )
 : LeafCandidate( ( char ) 0, p4 ),
   theDTMatch ( muRef ) ,
   theIsolation ( tkisol )
{
	// need to set the z of the track

}


	// Naive TkMuons
L1TkMuonParticle::L1TkMuonParticle( const LorentzVector& p4,
         const edm::Ref< L1MuonParticleCollection > &muRef,
         const edm::Ptr< L1TkTrackType >& trkPtr,
         float tkisol )
   : LeafCandidate( ( char ) 0, p4 ),
     muRef_ ( muRef ) ,	
     trkPtr_ ( trkPtr ) ,
     theIsolation ( tkisol )

{

 if ( trkPtr_.isNonnull() ) {
	float z = getTrkPtr() -> getPOCA().z();
	setTrkzVtx( z );
 }
}


int L1TkMuonParticle::bx() const {
 int dummy = -999;

 if (theDTMatch.isNonnull() ) {
   return theDTMatch->getDTBX();
 }

 if (muRef_.isNonnull() ) {
   return dummy;
 }

 return dummy;
}


unsigned int L1TkMuonParticle::quality() const {
 
 int dummy = 999;
 
 if ( muRef_.isNonnull() ) {
        return (muRef_ -> gmtMuonCand().quality() );
 }
 else {
        return dummy;
 }
}







