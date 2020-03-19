/*
 *  See header file for a description of this class.
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHK0sToPiPiBuilder.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticlePtSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticleEtaSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHMassSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHMassSymSelect.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHChi2Select.h"
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHParticleMasses.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"

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

BPHK0sToPiPiBuilder::BPHK0sToPiPiBuilder(const edm::EventSetup& es,
                                         const BPHRecoBuilder::BPHGenericCollection* posCollection,
                                         const BPHRecoBuilder::BPHGenericCollection* negCollection)
    : pionPosName("PionPos"),
      pionNegName("PionNeg"),
      evSetup(&es),
      pCollection(posCollection),
      nCollection(negCollection),
      vCollection(nullptr),
      rCollection(nullptr),
      sList("") {
  ptSel = new BPHParticlePtSelect(0.7);
  etaSel = new BPHParticleEtaSelect(10.0);
  massSel = new BPHMassSelect(0.40, 0.60);
  chi2Sel = new BPHChi2Select(0.0);
  updated = false;
}

BPHK0sToPiPiBuilder::BPHK0sToPiPiBuilder(const edm::EventSetup& es,
                                         const std::vector<reco::VertexCompositeCandidate>* v0Collection,
                                         const std::string& searchList)
    : pionPosName("PionPos"),
      pionNegName("PionNeg"),
      evSetup(&es),
      pCollection(nullptr),
      nCollection(nullptr),
      vCollection(v0Collection),
      rCollection(nullptr),
      sList(searchList) {
  ptSel = new BPHParticlePtSelect(0.0);
  etaSel = new BPHParticleEtaSelect(10.0);
  massSel = new BPHMassSelect(0.0, 2.0);
  chi2Sel = nullptr;
  updated = false;
}

BPHK0sToPiPiBuilder::BPHK0sToPiPiBuilder(const edm::EventSetup& es,
                                         const std::vector<reco::VertexCompositePtrCandidate>* vpCollection,
                                         const std::string& searchList)
    : pionPosName("PionPos"),
      pionNegName("PionNeg"),
      evSetup(&es),
      pCollection(nullptr),
      nCollection(nullptr),
      vCollection(nullptr),
      rCollection(vpCollection),
      sList(searchList) {
  ptSel = new BPHParticlePtSelect(0.0);
  etaSel = new BPHParticleEtaSelect(10.0);
  massSel = new BPHMassSelect(0.0, 2.0);
  chi2Sel = nullptr;
  updated = false;
}

//--------------
// Destructor --
//--------------
BPHK0sToPiPiBuilder::~BPHK0sToPiPiBuilder() {
  delete ptSel;
  delete etaSel;
  delete massSel;
  delete chi2Sel;
}

//--------------
// Operations --
//--------------
vector<BPHPlusMinusConstCandPtr> BPHK0sToPiPiBuilder::build() {
  if (updated)
    return k0sList;

  if ((pCollection != nullptr) && (nCollection != nullptr))
    buildFromBPHGenericCollection();
  else if (vCollection != nullptr)
    buildFromV0(vCollection);
  else if (rCollection != nullptr)
    buildFromV0(rCollection);

  updated = true;
  return k0sList;
}

void BPHK0sToPiPiBuilder::buildFromBPHGenericCollection() {
  k0sList.clear();

  BPHRecoBuilder bK0s(*evSetup);
  bK0s.add(pionPosName, pCollection, BPHParticleMasses::pionMass, BPHParticleMasses::pionMSigma);
  bK0s.add(pionNegName, nCollection, BPHParticleMasses::pionMass, BPHParticleMasses::pionMSigma);
  bK0s.filter(pionPosName, *ptSel);
  bK0s.filter(pionNegName, *ptSel);
  bK0s.filter(pionPosName, *etaSel);
  bK0s.filter(pionNegName, *etaSel);

  bK0s.filter(*massSel);
  if (chi2Sel != nullptr)
    bK0s.filter(*chi2Sel);

  k0sList = BPHPlusMinusCandidate::build(bK0s, pionPosName, pionNegName);

  return;
}

template <class T>
void BPHK0sToPiPiBuilder::buildFromV0(const T* v0Collection) {
  int iv0;
  int nv0 = v0Collection->size();
  k0sList.clear();
  k0sList.reserve(nv0);

  // cycle over std::vector<reco::VertexCompositeCandidate>* vCollection
  for (iv0 = 0; iv0 < nv0; ++iv0) {
    const typename T::value_type& kv0 = v0Collection->at(iv0);

    // every reco::VertexCompositeCandidate must have exactly two daughters
    if (kv0.numberOfDaughters() != 2)
      continue;

    // filters
    if (!ptSel->accept(*kv0.daughter(0)))
      continue;
    if (!etaSel->accept(*kv0.daughter(0)))
      continue;
    if (!ptSel->accept(*kv0.daughter(1)))
      continue;
    if (!etaSel->accept(*kv0.daughter(1)))
      continue;

    BPHPlusMinusCandidatePtr pk0 = BPHPlusMinusCandidateWrap::create(evSetup);
    BPHPlusMinusCandidate* k0s = pk0.get();

    // check which daughter has positive/negative charge
    if (kv0.daughter(0)->charge() > 0) {
      k0s->add(pionPosName, kv0.daughter(0), sList, BPHParticleMasses::pionMass);
      k0s->add(pionNegName, kv0.daughter(1), sList, BPHParticleMasses::pionMass);
    } else {
      k0s->add(pionPosName, kv0.daughter(1), sList, BPHParticleMasses::pionMass);
      k0s->add(pionNegName, kv0.daughter(0), sList, BPHParticleMasses::pionMass);
    }

    if (k0s->daughters().size() != 2)
      continue;
    if (!massSel->accept(*k0s))
      continue;
    if ((chi2Sel != nullptr) && (!chi2Sel->accept(*k0s)))
      continue;

    const_cast<pat::CompositeCandidate&>(k0s->composite()).addUserFloat("V0Mass", kv0.mass());
    k0sList.push_back(pk0);
  }

  return;
}

/// set cuts
void BPHK0sToPiPiBuilder::setPtMin(double pt) {
  updated = false;
  ptSel->setPtMin(pt);
  return;
}

void BPHK0sToPiPiBuilder::setEtaMax(double eta) {
  updated = false;
  etaSel->setEtaMax(eta);
  return;
}

void BPHK0sToPiPiBuilder::setMassMin(double m) {
  updated = false;
  massSel->setMassMin(m);
  return;
}

void BPHK0sToPiPiBuilder::setMassMax(double m) {
  updated = false;
  massSel->setMassMax(m);
  return;
}

void BPHK0sToPiPiBuilder::setProbMin(double p) {
  updated = false;
  delete chi2Sel;
  chi2Sel = (p < 0.0 ? nullptr : new BPHChi2Select(p));
  return;
}

void BPHK0sToPiPiBuilder::setConstr(double mass, double sigma) {
  updated = false;
  cMass = mass;
  cSigma = sigma;
  return;
}

/// get current cuts
double BPHK0sToPiPiBuilder::getPtMin() const { return ptSel->getPtMin(); }

double BPHK0sToPiPiBuilder::getEtaMax() const { return etaSel->getEtaMax(); }

double BPHK0sToPiPiBuilder::getMassMin() const { return massSel->getMassMin(); }

double BPHK0sToPiPiBuilder::getMassMax() const { return massSel->getMassMax(); }

double BPHK0sToPiPiBuilder::getProbMin() const { return (chi2Sel == nullptr ? -1.0 : chi2Sel->getProbMin()); }

double BPHK0sToPiPiBuilder::getConstrMass() const { return cMass; }

double BPHK0sToPiPiBuilder::getConstrSigma() const { return cSigma; }
