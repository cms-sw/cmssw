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
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHMuonChargeSelect.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "DataFormats/PatCandidates/interface/Muon.h"

//---------------
// C++ Headers --
//---------------


//-------------------
// Initializations --
//-------------------


//----------------
// Constructors --
//----------------
BPHMuonChargeSelect::BPHMuonChargeSelect( int c ): charge ( c ) {
}

//--------------
// Destructor --
//--------------
BPHMuonChargeSelect::~BPHMuonChargeSelect() {
}

//--------------
// Operations --
//--------------
bool BPHMuonChargeSelect::accept( const reco::Candidate& cand ) const {
  const pat::Muon* p = reinterpret_cast<const pat::Muon*>( &cand );
  if ( p == 0 ) return false;
  return ( ( charge * cand.charge() ) > 0 );
}

