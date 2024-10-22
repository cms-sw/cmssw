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
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayGenericBuilderBase.h"
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
BPHDecayToV0SameMassBuilder::BPHDecayToV0SameMassBuilder(const BPHEventSetupWrapper& es,
                                                         const string& posName,
                                                         const string& negName,
                                                         double daugMass,
                                                         double daugSigma,
                                                         const BPHRecoBuilder::BPHGenericCollection* posCollection,
                                                         const BPHRecoBuilder::BPHGenericCollection* negCollection)
    : BPHDecayGenericBuilderBase(es),
      BPHDecayToV0Builder(es, posName, negName, posCollection, negCollection),
      BPHDecayToChargedXXbarBuilder(es, posName, negName, daugMass, daugSigma, posCollection, negCollection),
      pMass(daugMass),
      pSigma(daugSigma) {}

BPHDecayToV0SameMassBuilder::BPHDecayToV0SameMassBuilder(const BPHEventSetupWrapper& es,
                                                         const string& posName,
                                                         const string& negName,
                                                         double daugMass,
                                                         double daugSigma,
                                                         const vector<reco::VertexCompositeCandidate>* v0Collection,
                                                         const string& searchList)
    : BPHDecayGenericBuilderBase(es),
      BPHDecayToV0Builder(es, posName, negName, v0Collection, searchList),
      BPHDecayToChargedXXbarBuilder(es, posName, negName, daugMass, daugSigma, nullptr, nullptr),
      pMass(daugMass),
      pSigma(daugSigma) {}

BPHDecayToV0SameMassBuilder::BPHDecayToV0SameMassBuilder(const BPHEventSetupWrapper& es,
                                                         const string& posName,
                                                         const string& negName,
                                                         double daugMass,
                                                         double daugSigma,
                                                         const vector<reco::VertexCompositePtrCandidate>* vpCollection,
                                                         const string& searchList)
    : BPHDecayGenericBuilderBase(es),
      BPHDecayToV0Builder(es, posName, negName, vpCollection, searchList),
      BPHDecayToChargedXXbarBuilder(es, posName, negName, daugMass, daugSigma, nullptr, nullptr),
      pMass(daugMass),
      pSigma(daugSigma) {}

//--------------
// Operations --
//--------------
void BPHDecayToV0SameMassBuilder::buildFromBPHGenericCollection() {
  BPHDecayToChargedXXbarBuilder::fillRecList();
  return;
}

BPHPlusMinusCandidatePtr BPHDecayToV0SameMassBuilder::buildCandidate(const reco::Candidate* c1,
                                                                     const reco::Candidate* c2,
                                                                     const void* v0,
                                                                     v0Type type) {
  if (c1->p4().pt() < ptMin)
    return nullptr;
  if (c2->p4().pt() < ptMin)
    return nullptr;
  if (fabs(c1->p4().eta()) > etaMax)
    return nullptr;
  if (fabs(c2->p4().eta()) > etaMax)
    return nullptr;
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
