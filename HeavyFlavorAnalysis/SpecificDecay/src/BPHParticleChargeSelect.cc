/*
 *  See header file for a description of this class.
 *
 *  $Date: 2016-05-03 14:57:34 $
 *  $Revision: 1.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticleChargeSelect.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"

//---------------
// C++ Headers --
//---------------


//-------------------
// Initializations --
//-------------------


//----------------
// Constructors --
//----------------
BPHParticleChargeSelect::BPHParticleChargeSelect( int c ): charge ( c ) {
}

//--------------
// Destructor --
//--------------
BPHParticleChargeSelect::~BPHParticleChargeSelect() {
}

//--------------
// Operations --
//--------------
bool BPHParticleChargeSelect::accept( const reco::Candidate& cand ) const {
  if ( !charge ) return !cand.charge();
  return ( ( ( charge > 0 ) && ( cand.charge() > 0 ) ) ||
           ( ( charge < 0 ) && ( cand.charge() < 0 ) ) );
}

