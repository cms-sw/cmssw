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
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayToTkpTknSymChargeBuilder.h"
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
BPHDecayToV0DiffMassBuilder::BPHDecayToV0DiffMassBuilder(const edm::EventSetup& es,
                                                         const std::string& d1Name,
                                                         double d1Mass,
                                                         double d1Sigma,
                                                         const std::string& d2Name,
                                                         double d2Mass,
                                                         double d2Sigma,
                                                         const BPHRecoBuilder::BPHGenericCollection* posCollection,
                                                         const BPHRecoBuilder::BPHGenericCollection* negCollection,
                                                         double expectedMass)
    : BPHDecayToV0Builder(es, d1Name, d2Name, posCollection, negCollection),
      p1Mass(d1Mass),
      p2Mass(d2Mass),
      p1Sigma(d1Sigma),
      p2Sigma(d2Sigma),
      expMass(expectedMass) {}

BPHDecayToV0DiffMassBuilder::BPHDecayToV0DiffMassBuilder(const edm::EventSetup& es,
                                                         const std::string& d1Name,
                                                         double d1Mass,
                                                         double d1Sigma,
                                                         const std::string& d2Name,
                                                         double d2Mass,
                                                         double d2Sigma,
                                                         const std::vector<reco::VertexCompositeCandidate>* v0Collection,
                                                         double expectedMass,
                                                         const std::string& searchList)
    : BPHDecayToV0Builder(es, d1Name, d2Name, v0Collection, searchList),
      p1Mass(d1Mass),
      p2Mass(d2Mass),
      p1Sigma(d1Sigma),
      p2Sigma(d2Sigma),
      expMass(expectedMass) {}

BPHDecayToV0DiffMassBuilder::BPHDecayToV0DiffMassBuilder(
    const edm::EventSetup& es,
    const std::string& d1Name,
    double d1Mass,
    double d1Sigma,
    const std::string& d2Name,
    double d2Mass,
    double d2Sigma,
    const std::vector<reco::VertexCompositePtrCandidate>* vpCollection,
    double expectedMass,
    const std::string& searchList)
    : BPHDecayToV0Builder(es, d1Name, d2Name, vpCollection, searchList),
      p1Mass(d1Mass),
      p2Mass(d2Mass),
      p1Sigma(d1Sigma),
      p2Sigma(d2Sigma),
      expMass(expectedMass) {}

//--------------
// Destructor --
//--------------
BPHDecayToV0DiffMassBuilder::~BPHDecayToV0DiffMassBuilder() {}

//--------------
// Operations --
//--------------
void BPHDecayToV0DiffMassBuilder::buildFromBPHGenericCollection() {
  BPHDecayToTkpTknSymChargeBuilder b(
      *evSetup, p1Name, p1Mass, p1Sigma, p2Name, p2Mass, p2Sigma, p1Collection, p2Collection, expMass);

  b.setTrk1PtMin(ptMin);
  b.setTrk2PtMin(ptMin);
  b.setTrk1EtaMax(etaMax);
  b.setTrk2EtaMax(etaMax);
  b.setMassRange(getMassMin(), getMassMax());
  b.setProbMin(getProbMin());

  cList = b.build();

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
