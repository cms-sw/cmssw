/*
 *  See header file for a description of this class.
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayToV0Builder.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHPlusMinusCandidate.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"

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
BPHDecayToV0Builder::BPHDecayToV0Builder(const edm::EventSetup& es,
                                         const std::string& d1Name,
                                         const std::string& d2Name,
                                         const BPHRecoBuilder::BPHGenericCollection* d1Collection,
                                         const BPHRecoBuilder::BPHGenericCollection* d2Collection)
    : BPHDecayGenericBuilder(es),
      p1Name(d1Name),
      p2Name(d2Name),
      p1Collection(d1Collection),
      p2Collection(d2Collection),
      vCollection(nullptr),
      rCollection(nullptr),
      sList(""),
      ptMin(0.0),
      etaMax(100.0) {}

BPHDecayToV0Builder::BPHDecayToV0Builder(const edm::EventSetup& es,
                                         const std::string& d1Name,
                                         const std::string& d2Name,
                                         const std::vector<reco::VertexCompositeCandidate>* v0Collection,
                                         const std::string& searchList)
    : BPHDecayGenericBuilder(es),
      p1Name(d1Name),
      p2Name(d2Name),
      p1Collection(nullptr),
      p2Collection(nullptr),
      vCollection(v0Collection),
      rCollection(nullptr),
      sList(searchList),
      ptMin(0.0),
      etaMax(100.0) {}

BPHDecayToV0Builder::BPHDecayToV0Builder(const edm::EventSetup& es,
                                         const std::string& d1Name,
                                         const std::string& d2Name,
                                         const std::vector<reco::VertexCompositePtrCandidate>* vpCollection,
                                         const std::string& searchList)
    : BPHDecayGenericBuilder(es),
      p1Name(d1Name),
      p2Name(d2Name),
      p1Collection(nullptr),
      p2Collection(nullptr),
      vCollection(nullptr),
      rCollection(vpCollection),
      sList(searchList),
      ptMin(0.0),
      etaMax(100.0) {}

//--------------
// Destructor --
//--------------
BPHDecayToV0Builder::~BPHDecayToV0Builder() { v0Clear(); }

//--------------
// Operations --
//--------------
vector<BPHPlusMinusConstCandPtr> BPHDecayToV0Builder::build() {
  if (updated)
    return cList;
  cList.clear();
  v0Clear();

  if ((p1Collection != nullptr) && (p2Collection != nullptr))
    buildFromBPHGenericCollection();
  else if (vCollection != nullptr)
    buildFromV0(vCollection, VertexCompositeCandidate);
  else if (rCollection != nullptr)
    buildFromV0(rCollection, VertexCompositePtrCandidate);

  updated = true;
  return cList;
}

/// set cuts
void BPHDecayToV0Builder::setPtMin(double pt) {
  updated = false;
  ptMin = pt;
  return;
}

void BPHDecayToV0Builder::setEtaMax(double eta) {
  updated = false;
  etaMax = eta;
  return;
}

/// get current cuts
double BPHDecayToV0Builder::getPtMin() const { return ptMin; }

double BPHDecayToV0Builder::getEtaMax() const { return etaMax; }

template <class T>
void BPHDecayToV0Builder::buildFromV0(const T* v0Collection, v0Type type) {
  int iv0;
  int nv0 = v0Collection->size();
  cList.reserve(nv0);

  // cycle over V0 collection
  for (iv0 = 0; iv0 < nv0; ++iv0) {
    const typename T::value_type& v0 = v0Collection->at(iv0);

    // every reco::VertexCompositeCandidate must have exactly two daughters
    if (v0.numberOfDaughters() != 2)
      continue;
    const reco::Candidate* dr = v0.daughter(0);
    const reco::Candidate* dl = v0.daughter(1);

    // filters
    if (dr->p4().pt() < ptMin)
      continue;
    if (dl->p4().pt() < ptMin)
      continue;
    if (fabs(dr->p4().eta()) > etaMax)
      continue;
    if (fabs(dl->p4().eta()) > etaMax)
      continue;

    BPHPlusMinusCandidatePtr cand = buildCandidate(dr, dl, &v0, type);
    BPHPlusMinusCandidate* cptr = cand.get();
    if (cand->daughters().size() != 2)
      continue;
    if (!massSel->accept(*cand))
      continue;
    if ((chi2Sel != nullptr) && (!chi2Sel->accept(*cand)))
      continue;

    cList.push_back(cand);
    V0Info* info = new V0Info;
    info->type = type;
    info->v0 = &v0;
    v0Map[cptr] = info;
  }

  return;
}

void BPHDecayToV0Builder::v0Clear() {
  map<const BPHRecoCandidate*, const V0Info*>::iterator iter = v0Map.begin();
  map<const BPHRecoCandidate*, const V0Info*>::iterator iend = v0Map.end();
  while (iter != iend)
    delete iter++->second;
  return;
}
