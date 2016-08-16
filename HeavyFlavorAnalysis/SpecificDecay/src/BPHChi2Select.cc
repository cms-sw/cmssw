/*
 *  See header file for a description of this class.
 *
 *  $Date: 2016-05-03 14:47:26 $
 *  $Revision: 1.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHChi2Select.h"


//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHDecayVertex.h"
#include "TMath.h"

//---------------
// C++ Headers --
//---------------


//-------------------
// Initializations --
//-------------------


//----------------
// Constructors --
//----------------
BPHChi2Select::BPHChi2Select( double prob ):
 probMin( prob ) {
}

//--------------
// Destructor --
//--------------
BPHChi2Select::~BPHChi2Select() {
}

//--------------
// Operations --
//--------------
/// select vertex
bool BPHChi2Select::accept( const BPHDecayVertex& cand ) const {
  const reco::Vertex& v = cand.vertex();
  if ( v.isFake() ) return false;
  if ( !v.isValid() ) return false;
  return ( TMath::Prob( v.chi2(), lround( v.ndof() ) ) > probMin );
}

/// set prob min
void BPHChi2Select::setProbMin( double p ) {
  probMin = p;
  return;
}

/// get current prob min
double BPHChi2Select::getProbMin() const {
  return probMin;
}

