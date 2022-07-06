/*
 *  See header file for a description of this class.
 *
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHDecayVertex.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHAnalyzerTokenWrapper.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoCandidate.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHRecoBuilder.h"
#include "HeavyFlavorAnalysis/RecoDecay/interface/BPHTrackReference.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
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
BPHDecayVertex::BPHDecayVertex(const BPHEventSetupWrapper* es, int daugNum, int compNum)
    : BPHDecayMomentum(daugNum, compNum),
      evSetup(new BPHEventSetupWrapper(es)),
      oldTracks(true),
      oldTTracks(true),
      oldVertex(true),
      validTks(false),
      savedFitter(nullptr),
      savedBS(nullptr),
      savedPP(nullptr),
      savedPE(nullptr) {}

BPHDecayVertex::BPHDecayVertex(const BPHDecayVertex* ptr, const BPHEventSetupWrapper* es)
    : evSetup(new BPHEventSetupWrapper(es)),
      oldTracks(true),
      oldTTracks(true),
      oldVertex(true),
      validTks(false),
      savedFitter(nullptr),
      savedBS(nullptr),
      savedPP(nullptr),
      savedPE(nullptr) {
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
    searchMap[iMap[c.cand]] = c.searchList;
  }
  const vector<BPHRecoConstCandPtr>& dComp = daughComp();
  int j;
  int m = dComp.size();
  for (j = 0; j < m; ++j) {
    const map<const reco::Candidate*, string>& dMap = dComp[j]->searchMap;
    searchMap.insert(dMap.begin(), dMap.end());
  }
}

//--------------
// Destructor --
//--------------
BPHDecayVertex::~BPHDecayVertex() { delete evSetup; }

//--------------
// Operations --
//--------------
bool BPHDecayVertex::validTracks() const {
  if (oldTracks)
    fTracks();
  return validTks;
}

bool BPHDecayVertex::validVertex() const {
  vertex();
  return validTks && fittedVertex.isValid();
}

const reco::Vertex& BPHDecayVertex::vertex(VertexFitter<5>* fitter,
                                           const reco::BeamSpot* bs,
                                           const GlobalPoint* priorPos,
                                           const GlobalError* priorError) const {
  if ((fitter == nullptr) && (bs == nullptr) && (priorPos == nullptr) && (priorError == nullptr)) {
    fitter = savedFitter;
    bs = savedBS;
    priorPos = savedPP;
    priorError = savedPE;
  }
  if (oldVertex || (fitter != savedFitter) || (bs != savedBS) || (priorPos != savedPP) || (priorError != savedPE)) {
    if (fitter != nullptr) {
      fitVertex(fitter, bs, priorPos, priorError);
    } else {
      KalmanVertexFitter kvf(true);
      fitVertex(&kvf, bs, priorPos, priorError);
    }
  }
  return fittedVertex;
}

const vector<const reco::Track*>& BPHDecayVertex::tracks() const {
  if (oldTracks)
    fTracks();
  return rTracks;
}

const reco::Track* BPHDecayVertex::getTrack(const reco::Candidate* cand) const {
  if (oldTracks)
    fTracks();
  map<const reco::Candidate*, const reco::Track*>::const_iterator iter = tkMap.find(cand);
  map<const reco::Candidate*, const reco::Track*>::const_iterator iend = tkMap.end();
  return (iter != iend ? iter->second : nullptr);
}

char BPHDecayVertex::getTMode(const reco::Candidate* cand) const {
  if (oldTracks)
    fTracks();
  map<const reco::Candidate*, char>::const_iterator iter = tmMap.find(cand);
  map<const reco::Candidate*, char>::const_iterator iend = tmMap.end();
  return (iter != iend ? iter->second : '.');
}

const vector<reco::TransientTrack>& BPHDecayVertex::transientTracks() const {
  if (oldTTracks)
    fTTracks();
  return trTracks;
}

reco::TransientTrack* BPHDecayVertex::getTransientTrack(const reco::Candidate* cand) const {
  if (oldTTracks)
    fTTracks();
  map<const reco::Candidate*, reco::TransientTrack*>::const_iterator iter = ttMap.find(cand);
  map<const reco::Candidate*, reco::TransientTrack*>::const_iterator iend = ttMap.end();
  return (iter != iend ? iter->second : nullptr);
}

/// retrieve EventSetup
const BPHEventSetupWrapper* BPHDecayVertex::getEventSetup() const { return evSetup; }

const string& BPHDecayVertex::getTrackSearchList(const reco::Candidate* cand) const {
  static const string dum = "";
  map<const reco::Candidate*, string>::const_iterator iter = searchMap.find(cand);
  if (iter != searchMap.end())
    return iter->second;
  return dum;
}

void BPHDecayVertex::addV(const string& name, const reco::Candidate* daug, const string& searchList, double mass) {
  addP(name, daug, mass);
  searchMap[daughters().back()] = searchList;
  return;
}

void BPHDecayVertex::addV(const string& name, const BPHRecoConstCandPtr& comp) {
  addP(name, comp);
  const map<const reco::Candidate*, string>& dMap = comp->searchMap;
  searchMap.insert(dMap.begin(), dMap.end());
  return;
}

void BPHDecayVertex::setNotUpdated() const {
  BPHDecayMomentum::setNotUpdated();
  oldTracks = oldVertex = true;
  validTks = false;
  return;
}

void BPHDecayVertex::fTracks() const {
  oldTTracks = true;
  rTracks.clear();
  tkMap.clear();
  tmMap.clear();
  const vector<const reco::Candidate*>& dL = daughFull();
  int n = dL.size();
  trTracks.reserve(n);
  validTks = true;
  while (n--) {
    const reco::Candidate* rp = dL[n];
    tkMap[rp] = nullptr;
    tmMap[rp] = '.';
    if (!rp->charge())
      continue;
    const char* searchList = "cfhp";
    char usedMode;
    map<const reco::Candidate*, string>::const_iterator iter = searchMap.find(rp);
    if (iter != searchMap.end())
      searchList = iter->second.c_str();
    const reco::Track* tp = tkMap[rp] = BPHTrackReference::getTrack(*originalReco(rp), searchList, &usedMode);
    if (tp == nullptr) {
      edm::LogPrint("DataNotFound") << "BPHDecayVertex::tTracks: "
                                    << "no track for reco::(PF)Candidate";
      validTks = false;
      continue;
    }
    rTracks.push_back(tp);
    tmMap[rp] = usedMode;
  }
  oldTracks = false;
  return;
}

void BPHDecayVertex::fTTracks() const {
  if (oldTracks)
    fTracks();
  trTracks.clear();
  BPHESTokenWrapper<TransientTrackBuilder, TransientTrackRecord>* token =
      evSetup->get<TransientTrackBuilder, TransientTrackRecord>(BPHRecoCandidate::transientTrackBuilder);
  const edm::EventSetup* ep = evSetup->get();
  edm::ESHandle<TransientTrackBuilder> ttB;
  token->get(*ep, ttB);
  ttMap.clear();
  const vector<const reco::Candidate*>& dL = daughFull();
  int n = dL.size();
  trTracks.reserve(n);
  while (n--) {
    const reco::Candidate* rp = dL[n];
    ttMap[rp] = nullptr;
    map<const reco::Candidate*, const reco::Track*>::const_iterator iter = tkMap.find(rp);
    const reco::Track* tp = iter->second;
    if (tp == nullptr)
      continue;
    trTracks.push_back(ttB->build(tp));
    ttMap[rp] = &trTracks.back();
  }
  oldTTracks = false;
}

void BPHDecayVertex::fitVertex(VertexFitter<5>* fitter,
                               const reco::BeamSpot* bs,
                               const GlobalPoint* priorPos,
                               const GlobalError* priorError) const {
  oldVertex = false;
  savedFitter = fitter;
  savedBS = bs;
  savedPP = priorPos;
  savedPE = priorError;
  if (oldTTracks)
    fTTracks();
  if (trTracks.size() < 2)
    return;
  try {
    if (bs == nullptr) {
      if (priorPos == nullptr) {
        TransientVertex tv = fitter->vertex(trTracks);
        fittedVertex = tv;
      } else {
        if (priorError == nullptr) {
          TransientVertex tv = fitter->vertex(trTracks, *priorPos);
          fittedVertex = tv;
        } else {
          TransientVertex tv = fitter->vertex(trTracks, *priorPos, *priorError);
          fittedVertex = tv;
        }
      }
    } else {
      TransientVertex tv = fitter->vertex(trTracks, *bs);
      fittedVertex = tv;
    }
  } catch (std::exception const&) {
    reco::Vertex tv;
    fittedVertex = tv;
    edm::LogPrint("FitFailed") << "BPHDecayVertex::fitVertex: "
                               << "vertex fit failed";
  }
  return;
}
