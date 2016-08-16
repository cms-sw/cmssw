/*
 *  See header file for a description of this class.
 *
 *  $Date: 2016-08-11 15:49:00 $
 *  $Revision: 1.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHMassFitSelect.h"


//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHKinematicFit.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicConstraint.h"
#include "RecoVertex/KinematicFitPrimitives/interface/MultiTrackKinematicConstraint.h"

//---------------
// C++ Headers --
//---------------
using namespace std;

//-------------------
// Initializations --
//-------------------


//----------------
// Constructors --
//----------------
BPHMassFitSelect::BPHMassFitSelect( double minMass, double maxMass ):
 cName ( ""   ),
 cMass ( -1.0 ),
 cSigma( -1.0 ),
   kc  (  0   ),
 mtkc  (  0   ),
 mMin( minMass ),
 mMax( maxMass ) {
}


BPHMassFitSelect::BPHMassFitSelect( const string& name,
                                    double mass, double sigma,
                                    double minMass, double maxMass ):
 cName ( name  ),
 cMass ( mass  ),
 cSigma( sigma ),
   kc  (  0   ),
 mtkc  (  0    ),
 mMin( minMass ),
 mMax( maxMass ) {
}


BPHMassFitSelect::BPHMassFitSelect( const string& name,
                                    double mass,
                                    double minMass, double maxMass ):
 cName ( name ),
 cMass ( mass ),
 cSigma( -1.0 ),
   kc  (  0   ),
 mtkc  (  0   ),
 mMin( minMass ),
 mMax( maxMass ) {
}


BPHMassFitSelect::BPHMassFitSelect( const string& name,
                                    KinematicConstraint* c,
                                    double minMass, double maxMass ):
 cName ( name ),
 cMass ( -1.0 ),
 cSigma( -1.0 ),
   kc  (  0   ),
 mtkc  (  0   ),
 mMin( minMass ),
 mMax( maxMass ) {
}


BPHMassFitSelect::BPHMassFitSelect( const string& name,
                                    MultiTrackKinematicConstraint* c,
                                    double minMass, double maxMass ):
 cName ( name ),
 cMass ( -1.0 ),
 cSigma( -1.0 ),
   kc  (  0   ),
 mtkc  (  c   ),
 mMin( minMass ),
 mMax( maxMass ) {
}

//--------------
// Destructor --
//--------------
BPHMassFitSelect::~BPHMassFitSelect() {
}

//--------------
// Operations --
//--------------
/// select particle
bool BPHMassFitSelect::accept( const BPHKinematicFit& cand ) const {
  if ( cMass > 0.0 ) cand.kinematicTree( cName, cMass, cSigma );
  if (   kc != 0   ) cand.kinematicTree( cName,   kc );
  if ( mtkc != 0   ) cand.kinematicTree( cName, mtkc );
  double mass = cand.p4().mass();
  return ( ( mass > mMin ) && ( mass < mMax ) );
}

/// set mass cuts
void BPHMassFitSelect::setMassMin( double m ) {
  mMin = m;
  return;
}


void BPHMassFitSelect::setMassMax( double m ) {
  mMax = m;
  return;
}

/// get current mass cuts
double BPHMassFitSelect::getMassMin() const {
  return mMin;
}


double BPHMassFitSelect::getMassMax() const {
  return mMax;
}

