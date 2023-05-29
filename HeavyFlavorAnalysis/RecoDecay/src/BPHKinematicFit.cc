/*
 *  See header file for a description of this class.
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHKinematicFit.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicParticleFactoryFromTransientTrack.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicParticle.h"
#include "RecoVertex/KinematicFit/interface/KinematicParticleVertexFitter.h"
#include "RecoVertex/KinematicFit/interface/KinematicConstrainedVertexFitter.h"
#include "RecoVertex/KinematicFit/interface/KinematicParticleFitter.h"
#include "RecoVertex/KinematicFit/interface/MassKinematicConstraint.h"
#include "RecoVertex/KinematicFit/interface/TwoTrackMassKinematicConstraint.h"
#include "RecoVertex/KinematicFit/interface/MultiTrackMassKinematicConstraint.h"
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
BPHKinematicFit::BPHKinematicFit(int daugNum, int compNum)
    : BPHDecayMomentum(daugNum, compNum),
      BPHDecayVertex(nullptr),
      massConst(-1.0),
      massSigma(-1.0),
      oldKPs(true),
      oldFit(true),
      oldMom(true),
      kinTree(nullptr) {}

BPHKinematicFit::BPHKinematicFit(const BPHKinematicFit* ptr)
    : BPHDecayVertex(ptr, nullptr),
      massConst(-1.0),
      massSigma(-1.0),
      oldKPs(true),
      oldFit(true),
      oldMom(true),
      kinTree(nullptr) {
  map<const reco::Candidate*, const reco::Candidate*> iMap;
  const vector<const reco::Candidate*>& daug = daughters();
  const vector<Component>& list = ptr->componentList();
  int i;
  int n = daug.size();
  for (i = 0; i < n; ++i) {
    const reco::Candidate* cand = daug[i];
    iMap[originalReco(cand)] = cand;
  }
  for (i = 0; i < n; ++i) {
    const Component& c = list[i];
    dMSig[iMap[c.cand]] = c.msig;
  }
  const vector<BPHRecoConstCandPtr>& dComp = daughComp();
  int j;
  int m = dComp.size();
  for (j = 0; j < m; ++j) {
    const BPHRecoCandidate* rc = dComp[j].get();
    const map<const reco::Candidate*, double>& dMap = rc->dMSig;
    const map<const BPHRecoCandidate*, FlyingParticle>& cMap = rc->cKinP;
    dMSig.insert(dMap.begin(), dMap.end());
    cKinP.insert(cMap.begin(), cMap.end());
    cKinP[rc];
  }
}

//--------------
// Operations --
//--------------
/// apply a mass constraint
void BPHKinematicFit::setConstraint(double mass, double sigma) {
  oldKPs = oldFit = oldMom = true;
  massConst = mass;
  massSigma = sigma;
  return;
}

/// retrieve the constraint
double BPHKinematicFit::constrMass() const { return massConst; }

double BPHKinematicFit::constrSigma() const { return massSigma; }

/// set a decaying daughter as an unique particle fitted independently
void BPHKinematicFit::setIndependentFit(const string& name, bool flag, double mass, double sigma) {
  string::size_type pos = name.find('/');
  if (pos != string::npos) {
    edm::LogPrint("WrongRequest") << "BPHKinematicFit::setIndependentFit: "
                                  << "cascade decay specification not admitted " << name;
    return;
  }
  const BPHRecoCandidate* comp = getComp(name).get();
  if (comp == nullptr) {
    edm::LogPrint("ParticleNotFound") << "BPHKinematicFit::setIndependentFit: " << name << " daughter not found";
    return;
  }
  oldKPs = oldFit = oldMom = true;
  FlyingParticle& fp = cKinP[comp];
  fp.flag = flag;
  fp.mass = mass;
  fp.sigma = sigma;
  return;
}

/// get kinematic particles
const vector<RefCountedKinematicParticle>& BPHKinematicFit::kinParticles() const {
  if (oldKPs)
    buildParticles();
  return allParticles;
}

vector<RefCountedKinematicParticle> BPHKinematicFit::kinParticles(const vector<string>& names) const {
  if (oldKPs)
    buildParticles();
  vector<RefCountedKinematicParticle> plist;
  unsigned int m = allParticles.size();
  if (m != numParticles())
    return plist;
  set<RefCountedKinematicParticle> pset;
  int i;
  int n = names.size();
  plist.reserve(m);
  for (i = 0; i < n; ++i) {
    const string& pname = names[i];
    if (pname == "*") {
      int j = m;
      while (j)
        insertParticle(allParticles[--j], plist, pset);
      break;
    }
    string::size_type pos = pname.find('/');
    if (pos != string::npos)
      getParticles(pname.substr(0, pos), pname.substr(pos + 1), plist, pset, this);
    else
      getParticles(pname, "", plist, pset, this);
  }
  return plist;
}

/// perform the kinematic fit and get the result
const RefCountedKinematicTree& BPHKinematicFit::kinematicTree() const {
  if (oldFit)
    return kinematicTree("", massConst, massSigma);
  return kinTree;
}

const RefCountedKinematicTree& BPHKinematicFit::kinematicTree(const string& name, double mass, double sigma) const {
  if (mass < 0)
    return kinematicTree(name);
  if (sigma < 0)
    return kinematicTree(name, mass);
  ParticleMass mc = mass;
  MassKinematicConstraint kinConst(mc, sigma);
  return kinematicTree(name, &kinConst);
}

const RefCountedKinematicTree& BPHKinematicFit::kinematicTree(const string& name, double mass) const {
  if (mass < 0)
    return kinematicTree(name);
  vector<RefCountedKinematicParticle> kPart;
  const BPHKinematicFit* ptr = splitKP(name, &kPart);
  if (ptr == nullptr) {
    edm::LogPrint("ParticleNotFound") << "BPHKinematicFit::kinematicTree: " << name << " daughter not found";
    return kinTree;
  }
  int nn = ptr->daughFull().size();
  ParticleMass mc = mass;
  if (nn == 2) {
    TwoTrackMassKinematicConstraint kinConst(mc);
    return kinematicTree(kPart, &kinConst);
  } else {
    MultiTrackMassKinematicConstraint kinConst(mc, nn);
    return kinematicTree(kPart, &kinConst);
  }
}

const RefCountedKinematicTree& BPHKinematicFit::kinematicTree(const string& name) const {
  KinematicConstraint* kc = nullptr;
  return kinematicTree(name, kc);
}

const RefCountedKinematicTree& BPHKinematicFit::kinematicTree(const string& name, KinematicConstraint* kc) const {
  vector<RefCountedKinematicParticle> kComp;
  vector<RefCountedKinematicParticle> kTail;
  const BPHKinematicFit* ptr = splitKP(name, &kComp, &kTail);
  if (ptr == nullptr) {
    edm::LogPrint("ParticleNotFound") << "BPHKinematicFit::kinematicTree: " << name << " daughter not found";
    return kinTree;
  }
  try {
    KinematicParticleVertexFitter vtxFitter;
    RefCountedKinematicTree compTree = vtxFitter.fit(kComp);
    if (compTree->isEmpty())
      return kinTree;
    if (kc != nullptr) {
      KinematicParticleFitter kinFitter;
      compTree = kinFitter.fit(kc, compTree);
      if (compTree->isEmpty())
        return kinTree;
    }
    compTree->movePointerToTheTop();
    if (!kTail.empty()) {
      RefCountedKinematicParticle compPart = compTree->currentParticle();
      if (!compPart->currentState().isValid())
        return kinTree;
      kTail.push_back(compPart);
      kinTree = vtxFitter.fit(kTail);
    } else {
      kinTree = compTree;
    }
  } catch (std::exception const&) {
    edm::LogPrint("FitFailed") << "BPHKinematicFit::kinematicTree: "
                               << "kin fit reset";
    kinTree = RefCountedKinematicTree(nullptr);
  }
  return kinTree;
}

const RefCountedKinematicTree& BPHKinematicFit::kinematicTree(const string& name,
                                                              MultiTrackKinematicConstraint* kc) const {
  vector<RefCountedKinematicParticle> kPart;
  if (splitKP(name, &kPart) == nullptr) {
    edm::LogPrint("ParticleNotFound") << "BPHKinematicFit::kinematicTree: " << name << " daughter not found";
    return kinTree;
  }
  return kinematicTree(kPart, kc);
}

/// reset the kinematic fit
void BPHKinematicFit::resetKinematicFit() const {
  oldKPs = oldFit = oldMom = true;
  return;
}

/// get fit status
bool BPHKinematicFit::isEmpty() const {
  kinematicTree();
  if (kinTree.get() == nullptr)
    return true;
  return kinTree->isEmpty();
}

bool BPHKinematicFit::isValidFit() const {
  const RefCountedKinematicParticle kPart = topParticle();
  if (kPart.get() == nullptr)
    return false;
  return kPart->currentState().isValid();
}

/// get current particle
const RefCountedKinematicParticle BPHKinematicFit::currentParticle() const {
  if (isEmpty())
    return RefCountedKinematicParticle(nullptr);
  return kinTree->currentParticle();
}

const RefCountedKinematicVertex BPHKinematicFit::currentDecayVertex() const {
  if (isEmpty())
    return RefCountedKinematicVertex(nullptr);
  return kinTree->currentDecayVertex();
}

/// get top particle
const RefCountedKinematicParticle BPHKinematicFit::topParticle() const {
  if (isEmpty())
    return RefCountedKinematicParticle(nullptr);
  return kinTree->topParticle();
}

const RefCountedKinematicVertex BPHKinematicFit::topDecayVertex() const {
  if (isEmpty())
    return RefCountedKinematicVertex(nullptr);
  kinTree->movePointerToTheTop();
  return kinTree->currentDecayVertex();
}

ParticleMass BPHKinematicFit::mass() const {
  const RefCountedKinematicParticle kPart = topParticle();
  if (kPart.get() == nullptr)
    return -1.0;
  const KinematicState kStat = kPart->currentState();
  if (kStat.isValid())
    return kStat.mass();
  return -1.0;
}

/// compute total momentum after the fit
const math::XYZTLorentzVector& BPHKinematicFit::p4() const {
  if (oldMom)
    fitMomentum();
  return totalMomentum;
}

/// retrieve particle mass sigma
double BPHKinematicFit::getMassSigma(const reco::Candidate* cand) const {
  map<const reco::Candidate*, double>::const_iterator iter = dMSig.find(cand);
  return (iter != dMSig.end() ? iter->second : -1);
}

/// retrieve independent fit flag
bool BPHKinematicFit::getIndependentFit(const string& name, double& mass, double& sigma) const {
  const BPHRecoCandidate* comp = getComp(name).get();
  map<const BPHRecoCandidate*, FlyingParticle>::const_iterator iter = cKinP.find(comp);
  if ((iter != cKinP.end()) && iter->second.flag) {
    mass = iter->second.mass;
    sigma = iter->second.sigma;
    return true;
  }
  return false;
}

/// add a simple particle and specify a criterion to search for
/// the associated track
void BPHKinematicFit::addK(const string& name, const reco::Candidate* daug, double mass, double sigma) {
  addK(name, daug, "cfhpmig", mass, sigma);
  return;
}

/// add a previously reconstructed particle giving it a name
void BPHKinematicFit::addK(
    const string& name, const reco::Candidate* daug, const string& searchList, double mass, double sigma) {
  addV(name, daug, searchList, mass);
  dMSig[daughters().back()] = sigma;
  return;
}

/// add a previously reconstructed particle giving it a name
void BPHKinematicFit::addK(const string& name, const BPHRecoConstCandPtr& comp) {
  addV(name, comp);
  const map<const reco::Candidate*, double>& dMap = comp->dMSig;
  const map<const BPHRecoCandidate*, FlyingParticle>& cMap = comp->cKinP;
  dMSig.insert(dMap.begin(), dMap.end());
  cKinP.insert(cMap.begin(), cMap.end());
  cKinP[comp.get()];
  return;
}

// utility function used to cash reconstruction results
void BPHKinematicFit::setNotUpdated() const {
  BPHDecayVertex::setNotUpdated();
  resetKinematicFit();
  return;
}

// build kin particles, perform the fit and compute the total momentum
void BPHKinematicFit::buildParticles() const {
  kinMap.clear();
  kCDMap.clear();
  allParticles.clear();
  allParticles.reserve(daughFull().size());
  addParticles(allParticles, kinMap, kCDMap);
  oldKPs = false;
  return;
}

void BPHKinematicFit::addParticles(vector<RefCountedKinematicParticle>& kl,
                                   map<const reco::Candidate*, RefCountedKinematicParticle>& km,
                                   map<const BPHRecoCandidate*, RefCountedKinematicParticle>& cm) const {
  const vector<const reco::Candidate*>& daug = daughters();
  KinematicParticleFactoryFromTransientTrack pFactory;
  int n = daug.size();
  float chi = 0.0;
  float ndf = 0.0;
  while (n--) {
    const reco::Candidate* cand = daug[n];
    ParticleMass mass = cand->mass();
    float sigma = dMSig.find(cand)->second;
    if (sigma < 0)
      sigma = 1.0e-7;
    reco::TransientTrack* tt = getTransientTrack(cand);
    if (tt != nullptr)
      kl.push_back(km[cand] = pFactory.particle(*tt, mass, chi, ndf, sigma));
  }
  const vector<BPHRecoConstCandPtr>& comp = daughComp();
  int m = comp.size();
  while (m--) {
    const BPHRecoCandidate* cptr = comp[m].get();
    const FlyingParticle& fp = cKinP.at(cptr);
    if (fp.flag) {
      BPHRecoCandidate* tptr = cptr->clone();
      double mass = fp.mass;
      double sigma = fp.sigma;
      if (mass > 0.0)
        tptr->setConstraint(mass, sigma);
      tmpList.push_back(BPHRecoConstCandPtr(tptr));
      if (tptr->isEmpty())
        return;
      if (!tptr->isValidFit())
        return;
      kl.push_back(cm[cptr] = tptr->topParticle());
    } else {
      cptr->addParticles(kl, km, cm);
    }
  }
  return;
}

void BPHKinematicFit::getParticles(const string& moth,
                                   const string& daug,
                                   vector<RefCountedKinematicParticle>& kl,
                                   set<RefCountedKinematicParticle>& ks,
                                   const BPHKinematicFit* curr) const {
  const BPHRecoCandidate* cptr = getComp(moth).get();
  if (cptr != nullptr) {
    if (curr->cKinP.at(cptr).flag) {
      insertParticle(kCDMap[cptr], kl, ks);
    } else {
      vector<string> list;
      if (!daug.empty()) {
        list.push_back(daug);
      } else {
        const vector<string>& dNames = cptr->daugNames();
        const vector<string>& cNames = cptr->compNames();
        list.insert(list.end(), dNames.begin(), dNames.end());
        list.insert(list.end(), cNames.begin(), cNames.end());
      }
      getParticles(moth, list, kl, ks, cptr);
    }
    return;
  }
  const reco::Candidate* dptr = getDaug(moth);
  if (dptr != nullptr) {
    insertParticle(kinMap[dptr], kl, ks);
    return;
  }
  edm::LogPrint("ParticleNotFound") << "BPHKinematicFit::getParticles: " << moth << " not found";
  return;
}

void BPHKinematicFit::getParticles(const string& moth,
                                   const vector<string>& daug,
                                   vector<RefCountedKinematicParticle>& kl,
                                   set<RefCountedKinematicParticle>& ks,
                                   const BPHKinematicFit* curr) const {
  int i;
  int n = daug.size();
  for (i = 0; i < n; ++i) {
    const string& name = daug[i];
    string::size_type pos = name.find('/');
    if (pos != string::npos)
      getParticles(moth + "/" + name.substr(0, pos), name.substr(pos + 1), kl, ks, curr);
    else
      getParticles(moth + "/" + name, "", kl, ks, curr);
  }
  return;
}

unsigned int BPHKinematicFit::numParticles(const BPHKinematicFit* cand) const {
  if (cand == nullptr)
    cand = this;
  unsigned int n = cand->daughters().size();
  const vector<string>& cnames = cand->compNames();
  int i = cnames.size();
  while (i) {
    const BPHRecoCandidate* comp = cand->getComp(cnames[--i]).get();
    if (cand->cKinP.at(comp).flag)
      ++n;
    else
      n += numParticles(comp);
  }
  return n;
}

void BPHKinematicFit::insertParticle(RefCountedKinematicParticle& kp,
                                     vector<RefCountedKinematicParticle>& kl,
                                     set<RefCountedKinematicParticle>& ks) {
  if (ks.find(kp) != ks.end())
    return;
  kl.push_back(kp);
  ks.insert(kp);
  return;
}

const BPHKinematicFit* BPHKinematicFit::splitKP(const string& name,
                                                vector<RefCountedKinematicParticle>* kComp,
                                                vector<RefCountedKinematicParticle>* kTail) const {
  kinTree = RefCountedKinematicTree(nullptr);
  oldFit = false;
  kinParticles();
  if (allParticles.size() != numParticles())
    return nullptr;
  kComp->clear();
  if (kTail == nullptr)
    kTail = kComp;
  else
    kTail->clear();
  if (name.empty()) {
    *kComp = allParticles;
    return this;
  }
  const BPHRecoCandidate* comp = getComp(name).get();
  int ns;
  if (comp != nullptr) {
    ns = (cKinP.at(comp).flag ? 1 : numParticles(comp));
  } else {
    edm::LogPrint("ParticleNotFound") << "BPHKinematicFit::splitKP: " << name << " daughter not found";
    *kTail = allParticles;
    return nullptr;
  }
  vector<string> nfull(2);
  nfull[0] = name;
  nfull[1] = "*";
  vector<RefCountedKinematicParticle> kPart = kinParticles(nfull);
  vector<RefCountedKinematicParticle>::const_iterator iter = kPart.begin();
  vector<RefCountedKinematicParticle>::const_iterator imid = iter + ns;
  vector<RefCountedKinematicParticle>::const_iterator iend = kPart.end();
  kComp->insert(kComp->end(), iter, imid);
  kTail->insert(kTail->end(), imid, iend);
  return comp;
}

const RefCountedKinematicTree& BPHKinematicFit::kinematicTree(const vector<RefCountedKinematicParticle>& kPart,
                                                              MultiTrackKinematicConstraint* kc) const {
  try {
    KinematicConstrainedVertexFitter cvf;
    kinTree = cvf.fit(kPart, kc);
  } catch (std::exception const&) {
    edm::LogPrint("FitFailed") << "BPHKinematicFit::kinematicTree: "
                               << "kin fit reset";
    kinTree = RefCountedKinematicTree(nullptr);
  }
  return kinTree;
}

void BPHKinematicFit::fitMomentum() const {
  if (isValidFit()) {
    const KinematicState& ks = topParticle()->currentState();
    GlobalVector tm = ks.globalMomentum();
    double x = tm.x();
    double y = tm.y();
    double z = tm.z();
    double m = ks.mass();
    double e = sqrt((x * x) + (y * y) + (z * z) + (m * m));
    totalMomentum.SetPxPyPzE(x, y, z, e);
  } else {
    edm::LogPrint("FitNotFound") << "BPHKinematicFit::fitMomentum: "
                                 << "simple momentum sum computed";
    math::XYZTLorentzVector tm;
    const vector<const reco::Candidate*>& daug = daughters();
    int n = daug.size();
    while (n--)
      tm += daug[n]->p4();
    const vector<BPHRecoConstCandPtr>& comp = daughComp();
    int m = comp.size();
    while (m--)
      tm += comp[m]->p4();
    totalMomentum = tm;
  }
  oldMom = false;
  return;
}
