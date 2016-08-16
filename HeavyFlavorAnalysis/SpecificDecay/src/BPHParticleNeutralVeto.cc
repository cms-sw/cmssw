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
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticleNeutralVeto.h"

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
BPHParticleNeutralVeto::BPHParticleNeutralVeto() {
}

//--------------
// Destructor --
//--------------
BPHParticleNeutralVeto::~BPHParticleNeutralVeto() {
}

//--------------
// Operations --
//--------------
bool BPHParticleNeutralVeto::accept( const reco::Candidate& cand ) const {
  return lround( fabs( cand.charge() ) );
}

