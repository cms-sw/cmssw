/*
 *  See header file for a description of this class.
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHDecayMomentum.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHAddFourMomenta.h"
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
BPHDecayMomentum::BPHDecayMomentum(int daugNum, int compNum) : oldMom(true) {
  nList.reserve(daugNum);
  dList.reserve(daugNum);
  nComp.reserve(compNum);
  cList.reserve(compNum);
}

BPHDecayMomentum::BPHDecayMomentum(const map<string, BPHDecayMomentum::Component>& daugMap, int compNum)
    : oldMom(true) {
  // clone and store simple particles
  clonesList(daugMap);
  nComp.reserve(compNum);
  cList.reserve(compNum);
}

BPHDecayMomentum::BPHDecayMomentum(const map<string, BPHDecayMomentum::Component>& daugMap,
                                   const map<string, BPHRecoConstCandPtr> compMap)
    :  // store the map of names to previously reconstructed particles
      cMap(compMap),
      oldMom(true) {
  // clone and store simple particles
  clonesList(daugMap);
  // store previously reconstructed particles using information in cMap
  dCompList();
}

//--------------
// Destructor --
//--------------
BPHDecayMomentum::~BPHDecayMomentum() {
  // delete all clones
  int n = dList.size();
  while (n--)
    delete dList[n];
}

//--------------
// Operations --
//--------------
const pat::CompositeCandidate& BPHDecayMomentum::composite() const {
  if (oldMom)
    computeMomentum();
  return compCand;
}

const vector<string>& BPHDecayMomentum::daugNames() const { return nList; }

const vector<string>& BPHDecayMomentum::compNames() const { return nComp; }

const vector<const reco::Candidate*>& BPHDecayMomentum::daughters() const { return dList; }

const vector<const reco::Candidate*>& BPHDecayMomentum::daughFull() const {
  // compute total momentum to update the full list of decay products
  if (oldMom)
    computeMomentum();
  return dFull;
}

const reco::Candidate* BPHDecayMomentum::originalReco(const reco::Candidate* daug) const {
  // return the original particle for a given clone
  // return null pointer if not found
  map<const reco::Candidate*, const reco::Candidate*>::const_iterator iter = clonesMap.find(daug);
  return (iter != clonesMap.end() ? iter->second : nullptr);
}

const vector<BPHRecoConstCandPtr>& BPHDecayMomentum::daughComp() const {
  // return the list of previously reconstructed particles
  return cList;
}

const reco::Candidate* BPHDecayMomentum::getDaug(const string& name) const {
  // return a simple particle from the name
  // return null pointer if not found
  string::size_type pos = name.find('/');
  if (pos != string::npos) {
    const BPHRecoCandidate* comp = getComp(name.substr(0, pos)).get();
    return (comp == nullptr ? nullptr : comp->getDaug(name.substr(pos + 1)));
  }
  map<const string, const reco::Candidate*>::const_iterator iter = dMap.find(name);
  return (iter != dMap.end() ? iter->second : nullptr);
}

BPHRecoConstCandPtr BPHDecayMomentum::getComp(const string& name) const {
  // return a previously reconstructed particle from the name
  // return null pointer if not found
  string::size_type pos = name.find('/');
  if (pos != string::npos) {
    const BPHRecoCandidate* comp = getComp(name.substr(0, pos)).get();
    return (comp == nullptr ? nullptr : comp->getComp(name.substr(pos + 1)));
  }
  map<const string, BPHRecoConstCandPtr>::const_iterator iter = cMap.find(name);
  return (iter != cMap.end() ? iter->second : nullptr);
}

const vector<BPHDecayMomentum::Component>& BPHDecayMomentum::componentList() const {
  // return an object filled in the constructor
  // to be used in the creation of other bases of BPHRecoCandidate
  return compList;
}

void BPHDecayMomentum::addP(const string& name, const reco::Candidate* daug, double mass) {
  if (dMap.find(name) != dMap.end()) {
    edm::LogPrint("TooManyParticles") << "BPHDecayMomentum::add: "
                                      << "Decay product already inserted with name " << name << " , skipped";
  }
  setNotUpdated();
  reco::Candidate* dnew = daug->clone();
  if (mass > 0.0)
    dnew->setMass(mass);
  nList.push_back(name);
  dList.push_back(dnew);
  dMap[name] = dnew;
  clonesMap[dnew] = daug;
  return;
}

void BPHDecayMomentum::addP(const string& name, const BPHRecoConstCandPtr& comp) {
  setNotUpdated();
  nComp.push_back(name);
  cList.push_back(comp);
  cMap[name] = comp;
  clonesMap.insert(comp->clonesMap.begin(), comp->clonesMap.end());
  return;
}

void BPHDecayMomentum::setNotUpdated() const {
  oldMom = true;
  return;
}

void BPHDecayMomentum::clonesList(const map<string, Component>& daugMap) {
  int n = daugMap.size();
  dList.resize(n);
  nList.resize(n);
  // reset and fill a list
  // to be used in the creation of other bases of BPHRecoCandidate
  compList.clear();
  compList.reserve(n);
  // loop over daughters
  int i = 0;
  double mass;
  reco::Candidate* dnew;
  map<string, Component>::const_iterator iter = daugMap.begin();
  map<string, Component>::const_iterator iend = daugMap.end();
  while (iter != iend) {
    const map<string, Component>::value_type& entry = *iter++;
    const Component& comp = entry.second;
    const reco::Candidate* cand = comp.cand;
    // store component for usage
    // in the creation of other bases of BPHRecoCandidate
    compList.push_back(comp);
    // clone particle and store it with its name
    dList[i] = dnew = cand->clone();
    const string& name = nList[i++] = entry.first;
    dMap[name] = dnew;
    clonesMap[dnew] = cand;
    // set daughter mass if requested
    mass = comp.mass;
    if (mass > 0)
      dnew->setMass(mass);
  }
  return;
}

void BPHDecayMomentum::dCompList() {
  // fill lists of previously reconstructed particles and their names
  // and retrieve cascade decay products
  int n = cMap.size();
  cList.resize(n);
  nComp.resize(n);
  int i = 0;
  map<string, BPHRecoConstCandPtr>::const_iterator c_iter = cMap.begin();
  map<string, BPHRecoConstCandPtr>::const_iterator c_iend = cMap.end();
  while (c_iter != c_iend) {
    const map<string, BPHRecoConstCandPtr>::value_type& c_entry = *c_iter++;
    nComp[i] = c_entry.first;
    BPHRecoConstCandPtr comp = c_entry.second;
    cList[i++] = comp;
    clonesMap.insert(comp->clonesMap.begin(), comp->clonesMap.end());
  }
  return;
}

void BPHDecayMomentum::sumMomentum(const vector<const reco::Candidate*>& dl, const vector<string>& dn) const {
  // add the particles to pat::CompositeCandidate
  int n = dl.size();
  while (n--)
    compCand.addDaughter(*dl[n], dn[n]);
  return;
}

void BPHDecayMomentum::fillDaug(vector<const reco::Candidate*>& ad, const string& name, vector<string>& an) const {
  // recursively fill the list of simple particles, produced
  // directly or in cascade decays
  ad.insert(ad.end(), dList.begin(), dList.end());
  vector<string>::const_iterator iter = nList.begin();
  vector<string>::const_iterator iend = nList.end();
  while (iter != iend)
    an.push_back(name + *iter++);
  int n = cList.size();
  while (n--)
    cList[n]->fillDaug(ad, name + nComp[n] + "/", an);
  return;
}

void BPHDecayMomentum::computeMomentum() const {
  // reset full list of daughters
  dFull.clear();
  dFull.reserve(10);
  vector<string> nFull;
  nFull.reserve(10);
  fillDaug(dFull, "", nFull);
  // reset and fill pat::CompositeCandidate
  compCand.clearDaughters();
  sumMomentum(dFull, nFull);
  // compute the total momentum
  AddFourMomenta addP4;
  addP4.set(compCand);
  oldMom = false;
  return;
}
