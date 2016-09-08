/*
 *  See header file for a description of this class.
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>

using namespace std;

//-------------------
// Initializations --
//-------------------


//----------------
// Constructors --
//----------------
BPHRecoCandidate::BPHRecoCandidate( const edm::EventSetup* es ):
  BPHDecayVertex( es ) {
}


BPHRecoCandidate::BPHRecoCandidate( const edm::EventSetup* es,
                  const BPHRecoBuilder::ComponentSet& compList ):
  BPHDecayMomentum( compList.daugMap, compList.compMap ),
  BPHDecayVertex( this, es ),
  BPHKinematicFit( this ) {
}

//--------------
// Destructor --
//--------------
BPHRecoCandidate::~BPHRecoCandidate() {
}


//--------------
// Operations --
//--------------
vector<BPHRecoConstCandPtr> BPHRecoCandidate::build(
                            const BPHRecoBuilder& builder,
                            double mass, double msig ) {
  // create a list of pointers to BPHRecoCandidate and fill it
  // with particle combinations selected by the BPHRecoBuilder
  vector<BPHRecoConstCandPtr> cList;
  fill<BPHRecoCandidate>( cList, builder, mass, msig );
  return cList;
}

