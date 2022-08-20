/*
 *  See header file for a description of this class.
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayToV0DiffMassBuilder.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayGenericBuilderBase.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayToTkpTknSymChargeBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"
#include "DataFormats/Candidate/interface/Candidate.h"

//---------------
// C++ Headers --
//---------------
#include <cmath>
using namespace std;

//-------------------
// Initializations --
//-------------------

//----------------
// Constructors --
//----------------
BPHDecayToV0DiffMassBuilder::BPHDecayToV0DiffMassBuilder(const BPHEventSetupWrapper& es,
                                                         const string& daug1Name,
                                                         double daug1Mass,
                                                         double daug1Sigma,
                                                         const string& daug2Name,
                                                         double daug2Mass,
                                                         double daug2Sigma,
                                                         const BPHRecoBuilder::BPHGenericCollection* daug1Collection,
                                                         const BPHRecoBuilder::BPHGenericCollection* daug2Collection,
                                                         double expectedMass)
    : BPHDecayGenericBuilderBase(es),
      BPHDecayToV0Builder(es, daug1Name, daug2Name, daug1Collection, daug2Collection),
      BPHDecayToTkpTknSymChargeBuilder(es,
                                       daug1Name,
                                       daug1Mass,
                                       daug1Sigma,
                                       daug2Name,
                                       daug2Mass,
                                       daug2Sigma,
                                       daug1Collection,
                                       daug2Collection,
                                       expectedMass),
      p1Mass(daug1Mass),
      p2Mass(daug2Mass),
      p1Sigma(daug1Sigma),
      p2Sigma(daug2Sigma),
      expMass(expectedMass) {}

BPHDecayToV0DiffMassBuilder::BPHDecayToV0DiffMassBuilder(const BPHEventSetupWrapper& es,
                                                         const string& daug1Name,
                                                         double daug1Mass,
                                                         double daug1Sigma,
                                                         const string& daug2Name,
                                                         double daug2Mass,
                                                         double daug2Sigma,
                                                         const vector<reco::VertexCompositeCandidate>* v0Collection,
                                                         double expectedMass,
                                                         const string& searchList)
    : BPHDecayGenericBuilderBase(es),
      BPHDecayToV0Builder(es, daug1Name, daug2Name, v0Collection, searchList),
      BPHDecayToTkpTknSymChargeBuilder(
          es, daug1Name, daug1Mass, daug1Sigma, daug2Name, daug2Mass, daug2Sigma, nullptr, nullptr, expectedMass),
      p1Mass(daug1Mass),
      p2Mass(daug2Mass),
      p1Sigma(daug1Sigma),
      p2Sigma(daug2Sigma),
      expMass(expectedMass) {}

BPHDecayToV0DiffMassBuilder::BPHDecayToV0DiffMassBuilder(const BPHEventSetupWrapper& es,
                                                         const string& daug1Name,
                                                         double daug1Mass,
                                                         double daug1Sigma,
                                                         const string& daug2Name,
                                                         double daug2Mass,
                                                         double daug2Sigma,
                                                         const vector<reco::VertexCompositePtrCandidate>* vpCollection,
                                                         double expectedMass,
                                                         const string& searchList)
    : BPHDecayGenericBuilderBase(es),
      BPHDecayToV0Builder(es, daug1Name, daug2Name, vpCollection, searchList),
      BPHDecayToTkpTknSymChargeBuilder(
          es, daug1Name, daug1Mass, daug1Sigma, daug2Name, daug2Mass, daug2Sigma, nullptr, nullptr, expectedMass),
      p1Mass(daug1Mass),
      p2Mass(daug2Mass),
      p1Sigma(daug1Sigma),
      p2Sigma(daug2Sigma),
      expMass(expectedMass) {}

//--------------
// Operations --
//--------------
void BPHDecayToV0DiffMassBuilder::buildFromBPHGenericCollection() {
  BPHDecayToTkpTknSymChargeBuilder::build();
  return;
}

BPHPlusMinusCandidatePtr BPHDecayToV0DiffMassBuilder::buildCandidate(const reco::Candidate* c1,
                                                                     const reco::Candidate* c2,
                                                                     const void* v0,
                                                                     v0Type type) {
  BPHPlusMinusCandidatePtr candX = BPHPlusMinusCandidateWrap::create(evSetup);
  BPHPlusMinusCandidatePtr candY = BPHPlusMinusCandidateWrap::create(evSetup);
  BPHPlusMinusCandidate* cptrX = candX.get();
  BPHPlusMinusCandidate* cptrY = candY.get();
  cptrX->add(p1Name, c1, sList, p1Mass, p1Sigma);
  cptrX->add(p2Name, c2, sList, p2Mass, p2Sigma);
  cptrY->add(p1Name, c2, sList, p1Mass, p1Sigma);
  cptrY->add(p2Name, c1, sList, p2Mass, p2Sigma);
  double mv0 = 0.0;
  switch (type) {
    case VertexCompositeCandidate:
      mv0 = static_cast<const reco::VertexCompositeCandidate*>(v0)->mass();
      break;
    case VertexCompositePtrCandidate:
      mv0 = static_cast<const reco::VertexCompositePtrCandidate*>(v0)->mass();
      break;
    default:
      mv0 = expMass;
      break;
  }
  double m1 = 0.0;
  double m2 = 0.0;
  if (p1Mass > p2Mass) {
    m1 = c1->mass();
    m2 = c2->mass();
  } else {
    m1 = c2->mass();
    m2 = c1->mass();
  }
  // check daughter masses in V0 CompositeCandidate
  double mcut = (p1Mass + p2Mass) / 2;
  if ((m1 > mcut) && (m2 < mcut))
    return candX;
  if ((m1 < mcut) && (m2 > mcut))
    return candY;
  // choose combination having the best invariant mass
  return (fabs(mv0 - cptrX->mass()) < fabs(mv0 - cptrY->mass()) ? candX : candY);
}
