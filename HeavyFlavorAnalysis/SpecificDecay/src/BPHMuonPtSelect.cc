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
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHMuonPtSelect.h"

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
BPHMuonPtSelect::BPHMuonPtSelect( double pt ): ptMin( pt ) {
}

//--------------
// Destructor --
//--------------
BPHMuonPtSelect::~BPHMuonPtSelect() {
}

//--------------
// Operations --
//--------------
/// select muon
bool BPHMuonPtSelect::accept( const reco::Candidate& cand ) const {
  const pat::Muon* p = reinterpret_cast<const pat::Muon*>( &cand );
  if ( p == 0 ) return false;
  return ( p->p4().pt() > ptMin );
}

/// set pt min
void BPHMuonPtSelect::setPtMin( double pt ) {
  ptMin = pt;
  return;
}

/// get current pt min
double BPHMuonPtSelect::getPtMin() const {
  return ptMin;
}

