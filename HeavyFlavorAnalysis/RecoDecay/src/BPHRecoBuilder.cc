/*
 *  See header file for a description of this class.
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHAnalyzerTokenWrapper.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoSelect.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHMomentumSelect.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHVertexSelect.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHFitSelect.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHTrackReference.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
BPHRecoBuilder::BPHRecoBuilder(const BPHEventSetupWrapper& es) : evSetup(new BPHEventSetupWrapper(es)), minPDiff(-1.0) {
  msList.reserve(5);
  vsList.reserve(5);
}

//--------------
// Destructor --
//--------------
BPHRecoBuilder::~BPHRecoBuilder() {
  int n = sourceList.size();
  while (n--) {
    delete sourceList[n]->collection;
    delete sourceList[n];
  }
  int m = srCompList.size();
  while (m--)
    delete srCompList[m];
  while (!compCollectList.empty()) {
    const vector<BPHRecoConstCandPtr>* cCollection = *compCollectList.begin();
    delete cCollection;
    compCollectList.erase(cCollection);
  }
  delete evSetup;
}

//--------------
// Operations --
//--------------
BPHRecoBuilder::BPHGenericCollection* BPHRecoBuilder::createCollection(const vector<const reco::Candidate*>& candList,
                                                                       const string& list) {
  return new BPHSpecificCollection<vector<const reco::Candidate*> >(candList, list);
}

void BPHRecoBuilder::add(const string& name, const BPHGenericCollection* collection, double mass, double msig) {
  BPHRecoSource* rs;
  if (sourceId.find(name) != sourceId.end()) {
    edm::LogPrint("TooManyParticles") << "BPHRecoBuilder::add: "
                                      << "Decay product already inserted with name " << name << " , skipped";
    return;
  }
  rs = new BPHRecoSource;
  rs->name = &sourceId.insert(make_pair(name, sourceList.size())).first->first;
  rs->collection = collection;
  rs->selector.reserve(5);
  rs->mass = mass;
  rs->msig = msig;
  sourceList.push_back(rs);
  return;
}

void BPHRecoBuilder::add(const string& name, const vector<BPHRecoConstCandPtr>& collection) {
  BPHCompSource* cs;
  if (srCompId.find(name) != srCompId.end()) {
    edm::LogPrint("TooManyParticles") << "BPHRecoBuilder::add: "
                                      << "Decay product already inserted with name " << name << " , skipped";
    return;
  }
  cs = new BPHCompSource;
  cs->name = &srCompId.insert(make_pair(name, srCompList.size())).first->first;
  cs->collection = &collection;
  srCompList.push_back(cs);
  return;
}

void BPHRecoBuilder::filter(const string& name, const BPHRecoSelect& sel) const {
  map<string, int>::const_iterator iter = sourceId.find(name);
  if (iter == sourceId.end())
    return;
  BPHRecoSource* rs = sourceList[iter->second];
  rs->selector.push_back(&sel);
  return;
}

void BPHRecoBuilder::filter(const string& name, const BPHMomentumSelect& sel) const {
  map<string, int>::const_iterator iter = srCompId.find(name);
  if (iter == sourceId.end())
    return;
  BPHCompSource* cs = srCompList[iter->second];
  cs->momSelector.push_back(&sel);
  return;
}

void BPHRecoBuilder::filter(const string& name, const BPHVertexSelect& sel) const {
  map<string, int>::const_iterator iter = srCompId.find(name);
  if (iter == sourceId.end())
    return;
  BPHCompSource* cs = srCompList[iter->second];
  cs->vtxSelector.push_back(&sel);
  return;
}

void BPHRecoBuilder::filter(const string& name, const BPHFitSelect& sel) const {
  map<string, int>::const_iterator iter = srCompId.find(name);
  if (iter == sourceId.end())
    return;
  BPHCompSource* cs = srCompList[iter->second];
  cs->fitSelector.push_back(&sel);
  return;
}

void BPHRecoBuilder::filter(const BPHMomentumSelect& sel) {
  msList.push_back(&sel);
  return;
}

void BPHRecoBuilder::filter(const BPHVertexSelect& sel) {
  vsList.push_back(&sel);
  return;
}

void BPHRecoBuilder::filter(const BPHFitSelect& sel) {
  fsList.push_back(&sel);
  return;
}

bool BPHRecoBuilder::accept(const BPHRecoCandidate& cand) const {
  int i;
  int n;
  n = msList.size();
  for (i = 0; i < n; ++i) {
    if (!msList[i]->accept(cand))
      return false;
  }
  n = vsList.size();
  for (i = 0; i < n; ++i) {
    if (!vsList[i]->accept(cand))
      return false;
  }
  n = fsList.size();
  for (i = 0; i < n; ++i) {
    if (!fsList[i]->accept(cand))
      return false;
  }
  return true;
}

void BPHRecoBuilder::setMinPDiffererence(double pMin) {
  minPDiff = pMin;
  return;
}

vector<BPHRecoBuilder::ComponentSet> BPHRecoBuilder::build() const {
  daugMap.clear();
  compMap.clear();
  vector<ComponentSet> candList;
  ComponentSet compSet;
  build(candList, compSet, sourceList.begin(), sourceList.end(), srCompList.begin(), srCompList.end());
  return candList;
}

const BPHEventSetupWrapper* BPHRecoBuilder::eventSetup() const { return evSetup; }

const reco::Candidate* BPHRecoBuilder::getDaug(const string& name) const {
  map<string, const reco::Candidate*>::const_iterator iter = daugMap.find(name);
  return (iter == daugMap.end() ? nullptr : iter->second);
}

BPHRecoConstCandPtr BPHRecoBuilder::getComp(const string& name) const {
  map<string, BPHRecoConstCandPtr>::const_iterator iter = compMap.find(name);
  return (iter == compMap.end() ? nullptr : iter->second);
}

bool BPHRecoBuilder::sameTrack(const reco::Candidate* lCand, const reco::Candidate* rCand, double minPDifference) {
  const reco::Track* lrcTrack = BPHTrackReference::getFromRC(*lCand);
  const reco::Track* rrcTrack = BPHTrackReference::getFromRC(*rCand);
  const reco::Track* lpfTrack = BPHTrackReference::getFromPF(*lCand);
  const reco::Track* rpfTrack = BPHTrackReference::getFromPF(*rCand);
  if ((lrcTrack != nullptr) && ((lrcTrack == rrcTrack) || (lrcTrack == rpfTrack)))
    return true;
  if ((lpfTrack != nullptr) && ((lpfTrack == rrcTrack) || (lpfTrack == rpfTrack)))
    return true;
  reco::Candidate::Vector pDiff = (lCand->momentum() - rCand->momentum());
  reco::Candidate::Vector pMean = (lCand->momentum() + rCand->momentum());
  double pDMod = pDiff.mag2();
  double pMMod = pMean.mag2();
  if (((pDMod / pMMod) < minPDifference) && (lCand->charge() == rCand->charge()))
    return true;
  return false;
}

void BPHRecoBuilder::build(vector<ComponentSet>& compList,
                           ComponentSet& compSet,
                           vector<BPHRecoSource*>::const_iterator r_iter,
                           vector<BPHRecoSource*>::const_iterator r_iend,
                           vector<BPHCompSource*>::const_iterator c_iter,
                           vector<BPHCompSource*>::const_iterator c_iend) const {
  if (r_iter == r_iend) {
    if (c_iter == c_iend) {
      compSet.compMap = compMap;
      compList.push_back(compSet);
      return;
    }
    BPHCompSource* source = *c_iter++;
    const vector<BPHRecoConstCandPtr>* collection = source->collection;
    vector<const BPHMomentumSelect*> momSelector = source->momSelector;
    vector<const BPHVertexSelect*> vtxSelector = source->vtxSelector;
    vector<const BPHFitSelect*> fitSelector = source->fitSelector;
    int i;
    int j;
    int n = collection->size();
    int m;
    bool skip;
    for (i = 0; i < n; ++i) {
      skip = false;
      BPHRecoConstCandPtr cand = collection->at(i);
      if (contained(compSet, cand))
        continue;
      m = momSelector.size();
      for (j = 0; j < m; ++j) {
        if (!momSelector[j]->accept(*cand, this)) {
          skip = true;
          break;
        }
      }
      if (skip)
        continue;
      m = vtxSelector.size();
      for (j = 0; j < m; ++j) {
        if (!vtxSelector[j]->accept(*cand, this)) {
          skip = true;
          break;
        }
      }
      if (skip)
        continue;
      m = fitSelector.size();
      for (j = 0; j < m; ++j) {
        if (!fitSelector[j]->accept(*cand, this)) {
          skip = true;
          break;
        }
      }
      if (skip)
        continue;
      compMap[*source->name] = cand;
      build(compList, compSet, r_iter, r_iend, c_iter, c_iend);
      compMap.erase(*source->name);
    }
    return;
  }
  BPHRecoSource* source = *r_iter++;
  const BPHGenericCollection* collection = source->collection;
  vector<const BPHRecoSelect*>& selector = source->selector;
  int i;
  int j;
  int n = collection->size();
  int m = selector.size();
  bool skip;
  for (i = 0; i < n; ++i) {
    const reco::Candidate& cand = collection->get(i);
    if (contained(compSet, &cand))
      continue;
    skip = false;
    for (j = 0; j < m; ++j) {
      if (!selector[j]->accept(cand, this)) {
        skip = true;
        break;
      }
    }
    if (skip)
      continue;
    BPHDecayMomentum::Component comp;
    comp.cand = &cand;
    comp.mass = source->mass;
    comp.msig = source->msig;
    comp.searchList = collection->searchList();
    compSet.daugMap[*source->name] = comp;
    daugMap[*source->name] = &cand;
    build(compList, compSet, r_iter, r_iend, c_iter, c_iend);
    daugMap.erase(*source->name);
    compSet.daugMap.erase(*source->name);
  }
  return;
}

bool BPHRecoBuilder::contained(ComponentSet& compSet, const reco::Candidate* cand) const {
  map<string, BPHDecayMomentum::Component>& dMap = compSet.daugMap;
  map<string, BPHDecayMomentum::Component>::const_iterator d_iter;
  map<string, BPHDecayMomentum::Component>::const_iterator d_iend = dMap.end();
  for (d_iter = dMap.begin(); d_iter != d_iend; ++d_iter) {
    const reco::Candidate* cChk = d_iter->second.cand;
    if (cand == cChk)
      return true;
    if (sameTrack(cand, cChk))
      return true;
  }
  return false;
}

bool BPHRecoBuilder::contained(ComponentSet& compSet, BPHRecoConstCandPtr cCand) const {
  map<string, BPHRecoConstCandPtr>::const_iterator c_iter;
  map<string, BPHRecoConstCandPtr>::const_iterator c_iend = compMap.end();
  const vector<const reco::Candidate*>& dCand = cCand->daughFull();
  int j;
  int m = dCand.size();
  int k;
  int l;
  for (j = 0; j < m; ++j) {
    const reco::Candidate* cand = cCand->originalReco(dCand[j]);
    map<string, BPHDecayMomentum::Component>& dMap = compSet.daugMap;
    map<string, BPHDecayMomentum::Component>::const_iterator d_iter;
    map<string, BPHDecayMomentum::Component>::const_iterator d_iend = dMap.end();
    for (d_iter = dMap.begin(); d_iter != d_iend; ++d_iter) {
      const reco::Candidate* cChk = d_iter->second.cand;
      if (cand == cChk)
        return true;
      if (sameTrack(cand, cChk))
        return true;
    }

    for (c_iter = compMap.begin(); c_iter != c_iend; ++c_iter) {
      const map<string, BPHRecoConstCandPtr>::value_type& entry = *c_iter;
      BPHRecoConstCandPtr cCChk = entry.second;
      const vector<const reco::Candidate*>& dCChk = cCChk->daughFull();
      l = dCChk.size();
      for (k = 0; k < l; ++k) {
        const reco::Candidate* cChk = cCChk->originalReco(dCChk[k]);
        if (cand == cChk)
          return true;
        if (sameTrack(cand, cChk))
          return true;
      }
    }
  }

  return false;
}

bool BPHRecoBuilder::sameTrack(const reco::Candidate* lCand, const reco::Candidate* rCand) const {
  return sameTrack(lCand, rCand, minPDiff);
}
