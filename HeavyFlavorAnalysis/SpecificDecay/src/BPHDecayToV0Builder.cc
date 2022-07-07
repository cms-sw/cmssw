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
#include "HeavyFlavorAnalysis/SpecificDecay/interface/BPHDecayGenericBuilderBase.h"
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
BPHDecayToV0Builder::BPHDecayToV0Builder(const BPHEventSetupWrapper& es,
                                         const string& d1Name,
                                         const string& d2Name,
                                         const BPHRecoBuilder::BPHGenericCollection* d1Collection,
                                         const BPHRecoBuilder::BPHGenericCollection* d2Collection)
    : BPHDecayGenericBuilderBase(es),
      p1Name(d1Name),
      p2Name(d2Name),
      p1Collection(d1Collection),
      p2Collection(d2Collection),
      vCollection(nullptr),
      rCollection(nullptr),
      sList("") {}

BPHDecayToV0Builder::BPHDecayToV0Builder(const BPHEventSetupWrapper& es,
                                         const string& d1Name,
                                         const string& d2Name,
                                         const vector<reco::VertexCompositeCandidate>* v0Collection,
                                         const string& searchList)
    : BPHDecayGenericBuilderBase(es),
      p1Name(d1Name),
      p2Name(d2Name),
      p1Collection(nullptr),
      p2Collection(nullptr),
      vCollection(v0Collection),
      rCollection(nullptr),
      sList(searchList) {}

BPHDecayToV0Builder::BPHDecayToV0Builder(const BPHEventSetupWrapper& es,
                                         const string& d1Name,
                                         const string& d2Name,
                                         const vector<reco::VertexCompositePtrCandidate>* vpCollection,
                                         const string& searchList)
    : BPHDecayGenericBuilderBase(es),
      p1Name(d1Name),
      p2Name(d2Name),
      p1Collection(nullptr),
      p2Collection(nullptr),
      vCollection(nullptr),
      rCollection(vpCollection),
      sList(searchList) {}

//--------------
// Destructor --
//--------------
BPHDecayToV0Builder::~BPHDecayToV0Builder() { v0Clear(); }

//--------------
// Operations --
//--------------
void BPHDecayToV0Builder::fillRecList() {
  v0Clear();

  if ((p1Collection != nullptr) && (p2Collection != nullptr))
    buildFromBPHGenericCollection();
  else if (vCollection != nullptr)
    buildFromV0(vCollection, VertexCompositeCandidate);
  else if (rCollection != nullptr)
    buildFromV0(rCollection, VertexCompositePtrCandidate);

  return;
}

template <class T>
void BPHDecayToV0Builder::buildFromV0(const T* v0Collection, v0Type type) {
  int iv0;
  int nv0 = v0Collection->size();
  recList.reserve(nv0);

  // cycle over V0 collection
  for (iv0 = 0; iv0 < nv0; ++iv0) {
    const typename T::value_type& v0 = v0Collection->at(iv0);

    // every reco::VertexCompositeCandidate must have exactly two daughters
    if (v0.numberOfDaughters() != 2)
      continue;
    const reco::Candidate* dr = v0.daughter(0);
    const reco::Candidate* dl = v0.daughter(1);

    // filters
    BPHPlusMinusCandidatePtr cand = buildCandidate(dr, dl, &v0, type);
    if (cand.get() == nullptr)
      continue;
    BPHPlusMinusCandidate* cptr = cand.get();
    if (cand->daughters().size() != 2)
      continue;
    if (!massSel->accept(*cand))
      continue;
    if ((chi2Sel != nullptr) && (!chi2Sel->accept(*cand)))
      continue;

    recList.push_back(cand);
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
