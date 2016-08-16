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
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticlePtSelect.h"

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
BPHParticlePtSelect::BPHParticlePtSelect( double pt ): ptMin( pt ) {
}

//--------------
// Destructor --
//--------------
BPHParticlePtSelect::~BPHParticlePtSelect() {
}

//--------------
// Operations --
//--------------
/// select particle
bool BPHParticlePtSelect::accept( const reco::Candidate& cand ) const {
  return ( cand.p4().pt() > ptMin );
}

/// set pt min
void BPHParticlePtSelect::setPtMin( double pt ) {
  ptMin = pt;
  return;
}

/// get current pt min
double BPHParticlePtSelect::getPtMin() const {
  return ptMin;
}

