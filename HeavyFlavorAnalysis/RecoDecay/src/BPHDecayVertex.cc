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
BPHDecayVertex::BPHDecayVertex(const edm::EventSetup* es)
    : evSetup(es),
      oldTracks(true),
      oldVertex(true),
      validTks(false),
      savedFitter(nullptr),
      savedBS(nullptr),
      savedPP(nullptr),
      savedPE(nullptr) {}

BPHDecayVertex::BPHDecayVertex(const BPHDecayVertex* ptr, const edm::EventSetup* es)
    : evSetup(es),
      oldTracks(true),
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
BPHDecayVertex::~BPHDecayVertex() {}

//--------------
// Operations --
//--------------
bool BPHDecayVertex::validTracks() const {
  if (oldTracks)
    tTracks();
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
    tTracks();
  return rTracks;
}

const reco::Track* BPHDecayVertex::getTrack(const reco::Candidate* cand) const {
  if (oldTracks)
    tTracks();
  map<const reco::Candidate*, const reco::Track*>::const_iterator iter = tkMap.find(cand);
  map<const reco::Candidate*, const reco::Track*>::const_iterator iend = tkMap.end();
  return (iter != iend ? iter->second : nullptr);
}

const vector<reco::TransientTrack>& BPHDecayVertex::transientTracks() const {
  if (oldTracks)
    tTracks();
  return trTracks;
}

reco::TransientTrack* BPHDecayVertex::getTransientTrack(const reco::Candidate* cand) const {
  if (oldTracks)
    tTracks();
  map<const reco::Candidate*, reco::TransientTrack*>::const_iterator iter = ttMap.find(cand);
  map<const reco::Candidate*, reco::TransientTrack*>::const_iterator iend = ttMap.end();
  return (iter != iend ? iter->second : nullptr);
}

/// retrieve EventSetup
const edm::EventSetup* BPHDecayVertex::getEventSetup() const { return evSetup; }

const string& BPHDecayVertex::getTrackSearchList(const reco::Candidate* cand) const {
  static string dum = "";
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

void BPHDecayVertex::tTracks() const {
  oldTracks = false;
  rTracks.clear();
  trTracks.clear();
  tkMap.clear();
  ttMap.clear();
  edm::ESHandle<TransientTrackBuilder> ttB;
  evSetup->get<TransientTrackRecord>().get("TransientTrackBuilder", ttB);
  const vector<const reco::Candidate*>& dL = daughFull();
  int n = dL.size();
  trTracks.reserve(n);
  validTks = true;
  while (n--) {
    const reco::Candidate* rp = dL[n];
    tkMap[rp] = nullptr;
    ttMap[rp] = nullptr;
    if (!rp->charge())
      continue;
    const reco::Track* tp;
    const char* searchList = "cfhp";
    map<const reco::Candidate*, string>::const_iterator iter = searchMap.find(rp);
    if (iter != searchMap.end())
      searchList = iter->second.c_str();
    tp = BPHTrackReference::getTrack(*originalReco(rp), searchList);
    if (tp == nullptr) {
      edm::LogPrint("DataNotFound") << "BPHDecayVertex::tTracks: "
                                    << "no track for reco::(PF)Candidate";
      validTks = false;
      continue;
    }
    rTracks.push_back(tp);
    trTracks.push_back(ttB->build(tp));
    reco::TransientTrack* ttp = &trTracks.back();
    tkMap[rp] = tp;
    ttMap[rp] = ttp;
  }
  return;
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
  if (oldTracks)
    tTracks();
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
