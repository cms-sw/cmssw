// -*- C++ -*-
//
// Package:     L1Trigger
// Class  :     L1TkMuonParticle

#include "DataFormats/L1TrackTrigger/interface/L1TkMuonParticle.h"

using namespace l1extra;

	// Padova's TkMuons
L1TkMuonParticle::L1TkMuonParticle( const LorentzVector& p4,
                                    const edm::Ptr< DTMatch > &muRef,
                                    float tkisol )
 : LeafCandidate( ( char ) 0, p4 ),
   theDTMatch ( muRef ) ,
   theIsolation ( tkisol ),
   TrkzVtx_(999.),
   quality_(999)
{
	// for DTMatch'es matched to a L1Track: set the reference
  	// to this L1Tracks and its zvertex
  const edm::Ptr< TTTrack< Ref_PixelDigi_ > >&  trkref = theDTMatch -> DTMatchBase::getPtMatchedTrackPtr();
  setTrkPtr( trkref );
  if ( trkPtr_.isNonnull() ) {
     float z = getTrkPtr() -> getPOCA().z();
     setTrkzVtx( z );
  }
}


	// Naive TkMuons
L1TkMuonParticle::L1TkMuonParticle( const LorentzVector& p4,
         const edm::Ref< L1MuonParticleCollection > &muRef,
         const edm::Ptr< L1TkTrackType >& trkPtr,
         float tkisol )
   : LeafCandidate( ( char ) 0, p4 ),
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
 if (muRef_.isNonnull()) {
   quality_ = muRef_->gmtMuonCand().quality();
 }
}


int L1TkMuonParticle::bx() const {
 // int dummy = -999;
 int dummy = 0;

/*
 if (theDTMatch.isNonnull() ) {
   return theDTMatch->getDTBX();
 }

 // PL.  In case Pierluigi's objects have a bx
 if ( muRef_.isNonnull() ) {
   return (getMuRef() -> bx()) ;
 }
 else if (muExtendedRef_.isNonnull()) {
   return getMuExtendedRef()->bx();
 
 }
*/

 return dummy;
}
