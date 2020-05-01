/*
 *  See header file for a description of this class.
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayToV0SameMassBuilder.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayToChargedXXbarBuilder.h"
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
BPHDecayToV0SameMassBuilder::BPHDecayToV0SameMassBuilder(const edm::EventSetup& es,
                                                         const std::string& d1Name,
                                                         const std::string& d2Name,
                                                         double dMass,
                                                         double dSigma,
                                                         const BPHRecoBuilder::BPHGenericCollection* d1Collection,
                                                         const BPHRecoBuilder::BPHGenericCollection* d2Collection)
    : BPHDecayToV0Builder(es, d1Name, d2Name, d1Collection, d2Collection), pMass(dMass), pSigma(dSigma) {}

BPHDecayToV0SameMassBuilder::BPHDecayToV0SameMassBuilder(const edm::EventSetup& es,
                                                         const std::string& d1Name,
                                                         const std::string& d2Name,
                                                         double dMass,
                                                         double dSigma,
                                                         const std::vector<reco::VertexCompositeCandidate>* v0Collection,
                                                         const std::string& searchList)
    : BPHDecayToV0Builder(es, d1Name, d2Name, v0Collection, searchList), pMass(dMass), pSigma(dSigma) {}

BPHDecayToV0SameMassBuilder::BPHDecayToV0SameMassBuilder(
    const edm::EventSetup& es,
    const std::string& d1Name,
    const std::string& d2Name,
    double dMass,
    double dSigma,
    const std::vector<reco::VertexCompositePtrCandidate>* vpCollection,
    const std::string& searchList)
    : BPHDecayToV0Builder(es, d1Name, d2Name, vpCollection, searchList), pMass(dMass), pSigma(dSigma) {}

//--------------
// Destructor --
//--------------
BPHDecayToV0SameMassBuilder::~BPHDecayToV0SameMassBuilder() {}

//--------------
// Operations --
//--------------
void BPHDecayToV0SameMassBuilder::buildFromBPHGenericCollection() {
  BPHDecayToChargedXXbarBuilder b(*evSetup, p1Name, p2Name, pMass, pSigma, p1Collection, p2Collection);

  b.setPtMin(ptMin);
  b.setEtaMax(etaMax);
  b.setMassRange(getMassMin(), getMassMax());
  b.setProbMin(getProbMin());

  cList = b.build();

  return;
}

BPHPlusMinusCandidatePtr BPHDecayToV0SameMassBuilder::buildCandidate(const reco::Candidate* c1,
                                                                     const reco::Candidate* c2,
                                                                     const void* v0,
                                                                     v0Type type) {
  BPHPlusMinusCandidatePtr cand = BPHPlusMinusCandidateWrap::create(evSetup);
  if (c1->charge() > 0) {
    cand->add(p1Name, c1, pMass, pSigma);
    cand->add(p2Name, c2, pMass, pSigma);
  } else {
    cand->add(p1Name, c2, pMass, pSigma);
    cand->add(p2Name, c1, pMass, pSigma);
  }
  return cand;
}
