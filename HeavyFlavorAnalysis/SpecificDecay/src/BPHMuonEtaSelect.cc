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
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHMuonEtaSelect.h"

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
BPHMuonEtaSelect::BPHMuonEtaSelect( double eta ): etaMax( eta ) {
}

//--------------
// Destructor --
//--------------
BPHMuonEtaSelect::~BPHMuonEtaSelect() {
}

//--------------
// Operations --
//--------------
/// select muon
bool BPHMuonEtaSelect::accept( const reco::Candidate& cand ) const {
  const pat::Muon* p = reinterpret_cast<const pat::Muon*>( &cand );
  if ( p == 0 ) return false;
  return ( fabs( p->p4().eta() ) < etaMax );
}

/// set eta max
void BPHMuonEtaSelect::setEtaMax( double eta ) {
  etaMax = eta;
  return;
}

/// get current eta max
double BPHMuonEtaSelect::getEtaMax() const {
  return etaMax;
}

