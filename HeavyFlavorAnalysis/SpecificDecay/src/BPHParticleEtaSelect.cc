/*
 *  See header file for a descrietaion of this class.
 *
 *  $Date: 2016-05-03 14:57:34 $
 *  $Revision: 1.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticleEtaSelect.h"

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
BPHParticleEtaSelect::BPHParticleEtaSelect( double eta ): etaMax( eta ) {
}

//--------------
// Destructor --
//--------------
BPHParticleEtaSelect::~BPHParticleEtaSelect() {
}

//--------------
// Operations --
//--------------
/// select particle
bool BPHParticleEtaSelect::accept( const reco::Candidate& cand ) const {
  return ( fabs( cand.p4().eta() ) < etaMax );
}

/// set eta max
void BPHParticleEtaSelect::setEtaMax( double eta ) {
  etaMax = eta;
  return;
}

/// get current eta max
double BPHParticleEtaSelect::getEtaMax() const {
  return etaMax;
}

