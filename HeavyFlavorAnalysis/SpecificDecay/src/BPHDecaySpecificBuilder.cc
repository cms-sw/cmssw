/*
 *  See header file for a description of this class.
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecaySpecificBuilder.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"

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
template <>
void BPHDecaySpecificBuilder<BPHRecoCandidate>::fillRecList() {
  BPHRecoBuilder brb(*evSetup);
  fill(brb, nullptr);
  this->recList = BPHRecoCandidate::build(brb);
  return;
}

template <>
void BPHDecaySpecificBuilder<BPHPlusMinusCandidate>::fillRecList() {
  BPHRecoBuilder brb(*evSetup);
  BPHPlusMinusCandidate::BuilderParameters buildPar;
  fill(brb, &buildPar);
  this->recList = BPHPlusMinusCandidate::build(brb, buildPar);
  return;
}
