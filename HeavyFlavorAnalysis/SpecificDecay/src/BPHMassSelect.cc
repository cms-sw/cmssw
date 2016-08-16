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
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHMassSelect.h"


//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHDecayMomentum.h"

//---------------
// C++ Headers --
//---------------


//-------------------
// Initializations --
//-------------------


//----------------
// Constructors --
//----------------
BPHMassSelect::BPHMassSelect( double minMass, double maxMass ):
 mMin( minMass ),
 mMax( maxMass ) {
}

//--------------
// Destructor --
//--------------
BPHMassSelect::~BPHMassSelect() {
}

//--------------
// Operations --
//--------------
/// select particle
bool BPHMassSelect::accept( const BPHDecayMomentum& cand ) const {
  double mass = cand.composite().mass();
  return ( ( mass > mMin ) && ( mass < mMax ) );
}

/// set mass cuts
void BPHMassSelect::setMassMin( double m ) {
  mMin = m;
  return;
}


void BPHMassSelect::setMassMax( double m ) {
  mMax = m;
  return;
}

/// get current mass cuts
double BPHMassSelect::getMassMin() const {
  return mMin;
}


double BPHMassSelect::getMassMax() const {
  return mMax;
}

