/*
 *  See header file for a description of this class.
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHMultiSelect.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoSelect.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHMomentumSelect.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHVertexSelect.h"

//---------------
// C++ Headers --
//---------------


//-------------------
// Initializations --
//-------------------


//----------------
// Constructors --
//----------------
// see interface/BPHMultiSelect.hpp

//--------------
// Destructor --
//--------------
// see interface/BPHMultiSelect.hpp

//--------------
// Operations --
//--------------
template<>
bool BPHMultiSelect<BPHRecoSelect    >::accept(
                                        const reco::Candidate & cand,
                                        const BPHRecoBuilder* build ) const {
  return select( cand, build );
}


template<>
bool BPHMultiSelect<BPHRecoSelect    >::accept(
                                        const reco::Candidate & cand ) const {
  return select( cand );
}


template<>
bool BPHMultiSelect<BPHMomentumSelect>::accept(
                                        const BPHDecayMomentum& cand ) const {
  return select( cand );
}


template<>
bool BPHMultiSelect<BPHVertexSelect  >::accept(
                                        const BPHDecayVertex  & cand ) const {
  return select( cand );
}


template<>
bool BPHMultiSelect<BPHFitSelect     >::accept(
                                        const BPHKinematicFit & cand ) const {
  return select( cand );
}

