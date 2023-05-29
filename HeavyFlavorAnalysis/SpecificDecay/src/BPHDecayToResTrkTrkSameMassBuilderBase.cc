/*
 *  See header file for a description of this class.
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayToResTrkTrkSameMassBuilderBase.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayGenericBuilderBase.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayToChargedXXbarBuilder.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticleNeutralVeto.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"

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
BPHDecayToResTrkTrkSameMassBuilderBase::BPHDecayToResTrkTrkSameMassBuilderBase(
    const BPHEventSetupWrapper& es,
    const string& resName,
    double resMass,
    double resWidth,
    const string& posName,
    const std::string& negName,
    double trkMass,
    double trkSigma,
    const BPHRecoBuilder::BPHGenericCollection* posCollection,
    const BPHRecoBuilder::BPHGenericCollection* negCollection)
    : BPHDecayGenericBuilderBase(es, nullptr),
      BPHDecayConstrainedBuilderBase(resName, resMass, resWidth),
      pName(posName),
      nName(negName),
      tMass(trkMass),
      tSigma(trkSigma),
      pCollection(posCollection),
      nCollection(negCollection),
      ptMin(0.0),
      etaMax(100.0) {}

BPHDecayToResTrkTrkSameMassBuilderBase::BPHDecayToResTrkTrkSameMassBuilderBase(
    const string& posName,
    const std::string& negName,
    double trkMass,
    double trkSigma,
    const BPHRecoBuilder::BPHGenericCollection* posCollection,
    const BPHRecoBuilder::BPHGenericCollection* negCollection)
    : pName(posName),
      nName(negName),
      tMass(trkMass),
      tSigma(trkSigma),
      pCollection(posCollection),
      nCollection(negCollection),
      ptMin(0.0),
      etaMax(100.0) {}

//--------------
// Operations --
//--------------
void BPHDecayToResTrkTrkSameMassBuilderBase::fillTrkTrkList() {
  double mTotMax = massSel->getMassMax();

  BPHDecayToChargedXXbarBuilder ttBuilder(*evSetup, pName, nName, tMass, tSigma, pCollection, nCollection);
  ttBuilder.setPtMin(ptMin);
  ttBuilder.setEtaMax(etaMax);
  ttBuilder.setDzMax(1.0);
  ttBuilder.setMassMin(0.0);
  if (mTotMax >= 0.0)
    ttBuilder.setMassMax(mTotMax - (0.8 * rMass));
  else
    ttBuilder.setMassMax(-1.0);
  ttBuilder.setMinPDiff(minPDiff);

  ttPairs = ttBuilder.build();

  return;
}

/// set cuts
void BPHDecayToResTrkTrkSameMassBuilderBase::setTrkPtMin(double pt) {
  outdated = true;
  ptMin = pt;
  return;
}

void BPHDecayToResTrkTrkSameMassBuilderBase::setTrkEtaMax(double eta) {
  outdated = true;
  etaMax = eta;
  return;
}
