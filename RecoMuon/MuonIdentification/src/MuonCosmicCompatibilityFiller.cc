/**
 *  \package: MuonIdentification
 *  \class: MuonCosmicCompatibilityFiller
 *
 *  Description: class for cosmic muon identification
 *
 *
 *  \author: A. Everett, Purdue University
 *  \author: A. Svyatkovskiy, Purdue University
 *  \author: H.D. Yoo, Purdue University
 *
 **/

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "DataFormats/DetId/interface/DetId.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"

#include "RecoMuon/MuonIdentification/interface/MuonCosmicCompatibilityFiller.h"
#include "RecoMuon/MuonIdentification/interface/MuonCosmicsId.h"

#include "DataFormats/MuonReco/interface/MuonCosmicCompatibility.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

#include "TMath.h"

using namespace edm;
using namespace std;

MuonCosmicCompatibilityFiller::MuonCosmicCompatibilityFiller(const edm::ParameterSet& iConfig,
                                                             edm::ConsumesCollector& iC)
    : inputMuonCollections_(iConfig.getParameter<std::vector<edm::InputTag> >("InputMuonCollections")),
      inputTrackCollections_(iConfig.getParameter<std::vector<edm::InputTag> >("InputTrackCollections")),
      inputCosmicMuonCollection_(iConfig.getParameter<edm::InputTag>("InputCosmicMuonCollection")),
      inputVertexCollection_(iConfig.getParameter<edm::InputTag>("InputVertexCollection")) {
  //kinematic vars
  angleThreshold_ = iConfig.getParameter<double>("angleCut");
  deltaPt_ = iConfig.getParameter<double>("deltaPt");
  //time
  offTimePosTightMult_ = iConfig.getParameter<double>("offTimePosTightMult");
  offTimeNegTightMult_ = iConfig.getParameter<double>("offTimeNegTightMult");
  offTimePosTight_ = iConfig.getParameter<double>("offTimePosTight");
  offTimeNegTight_ = iConfig.getParameter<double>("offTimeNegTight");
  offTimePosLooseMult_ = iConfig.getParameter<double>("offTimePosLooseMult");
  offTimeNegLooseMult_ = iConfig.getParameter<double>("offTimeNegLooseMult");
  offTimePosLoose_ = iConfig.getParameter<double>("offTimePosLoose");
  offTimeNegLoose_ = iConfig.getParameter<double>("offTimeNegLoose");
  corrTimeNeg_ = iConfig.getParameter<double>("corrTimeNeg");
  corrTimePos_ = iConfig.getParameter<double>("corrTimePos");
  //rechits
  sharedHits_ = iConfig.getParameter<int>("sharedHits");
  sharedFrac_ = iConfig.getParameter<double>("sharedFrac");
  ipThreshold_ = iConfig.getParameter<double>("ipCut");
  //segment comp, matches
  nChamberMatches_ = iConfig.getParameter<int>("nChamberMatches");
  segmentComp_ = iConfig.getParameter<double>("segmentComp");
  //ip, vertex
  maxdzLooseMult_ = iConfig.getParameter<double>("maxdzLooseMult");
  maxdxyLooseMult_ = iConfig.getParameter<double>("maxdxyLooseMult");
  maxdzTightMult_ = iConfig.getParameter<double>("maxdzTightMult");
  maxdxyTightMult_ = iConfig.getParameter<double>("maxdxyTightMult");
  maxdzLoose_ = iConfig.getParameter<double>("maxdzLoose");
  maxdxyLoose_ = iConfig.getParameter<double>("maxdxyLoose");
  maxdzTight_ = iConfig.getParameter<double>("maxdzTight");
  maxdxyTight_ = iConfig.getParameter<double>("maxdxyTight");
  largedxyMult_ = iConfig.getParameter<double>("largedxyMult");
  largedxy_ = iConfig.getParameter<double>("largedxy");
  hIpTrdxy_ = iConfig.getParameter<double>("hIpTrdxy");
  hIpTrvProb_ = iConfig.getParameter<double>("hIpTrvProb");
  minvProb_ = iConfig.getParameter<double>("minvProb");
  maxvertZ_ = iConfig.getParameter<double>("maxvertZ");
  maxvertRho_ = iConfig.getParameter<double>("maxvertRho");
  //  nTrackThreshold_ = iConfig.getParameter<unsigned int>("nTrackThreshold");

  for (unsigned int i = 0; i < inputMuonCollections_.size(); ++i)
    muonTokens_.push_back(iC.consumes<reco::MuonCollection>(inputMuonCollections_.at(i)));
  for (unsigned int i = 0; i < inputTrackCollections_.size(); ++i)
    trackTokens_.push_back(iC.consumes<reco::TrackCollection>(inputTrackCollections_.at(i)));

  cosmicToken_ = iC.consumes<reco::MuonCollection>(inputCosmicMuonCollection_);
  vertexToken_ = iC.consumes<reco::VertexCollection>(inputVertexCollection_);
  geometryToken_ = iC.esConsumes<GlobalTrackingGeometry, GlobalTrackingGeometryRecord>();
}

MuonCosmicCompatibilityFiller::~MuonCosmicCompatibilityFiller() {}

reco::MuonCosmicCompatibility MuonCosmicCompatibilityFiller::fillCompatibility(const reco::Muon& muon,
                                                                               edm::Event& iEvent,
                                                                               const edm::EventSetup& iSetup) {
  const std::string theCategory = "MuonCosmicCompatibilityFiller";

  reco::MuonCosmicCompatibility returnComp;

  float timeCompatibility = muonTiming(iEvent, muon, false);
  float backToBackCompatibility = backToBack2LegCosmic(iEvent, muon);
  float overlapCompatibility = isOverlappingMuon(iEvent, iSetup, muon);
  float ipCompatibility = pvMatches(iEvent, muon, false);
  float vertexCompatibility = eventActivity(iEvent, muon);
  float combinedCompatibility = combinedCosmicID(iEvent, iSetup, muon, false, false);

  returnComp.timeCompatibility = timeCompatibility;
  returnComp.backToBackCompatibility = backToBackCompatibility;
  returnComp.overlapCompatibility = overlapCompatibility;
  returnComp.cosmicCompatibility = combinedCompatibility;
  returnComp.ipCompatibility = ipCompatibility;
  returnComp.vertexCompatibility = vertexCompatibility;

  return returnComp;
}

//
//Timing: 0 - not cosmic-like
//
float MuonCosmicCompatibilityFiller::muonTiming(const edm::Event& iEvent, const reco::Muon& muon, bool isLoose) const {
  float offTimeNegMult, offTimePosMult, offTimeNeg, offTimePos;

  if (isLoose) {
    //use "loose" parameter set
    offTimeNegMult = offTimeNegLooseMult_;
    offTimePosMult = offTimePosLooseMult_;
    offTimeNeg = offTimeNegLoose_;
    offTimePos = offTimePosLoose_;
  } else {
    offTimeNegMult = offTimeNegTightMult_;
    offTimePosMult = offTimePosTightMult_;
    offTimeNeg = offTimeNegTight_;
    offTimePos = offTimePosTight_;
  }

  float result = 0.0;

  if (muon.isTimeValid()) {
    //case of multiple muon event
    if (nMuons(iEvent) > 1) {
      float positiveTime = 0;
      if (muon.time().timeAtIpInOut < offTimeNegMult || muon.time().timeAtIpInOut > offTimePosMult)
        result = 1.;
      if (muon.time().timeAtIpInOut > 0.)
        positiveTime = muon.time().timeAtIpInOut;

      //special case, looking for time-correlation
      // between muons in opposite hemispheres
      if (!isLoose && result == 0 && positiveTime > corrTimePos_) {
        //check hemi of this muon
        bool isUp = false;
        reco::TrackRef outertrack = muon.outerTrack();
        if (outertrack.isNonnull()) {
          if (outertrack->phi() > 0)
            isUp = true;

          //loop over muons in that event and find if there are any in the opposite hemi
          edm::Handle<reco::MuonCollection> muonHandle;
          iEvent.getByToken(muonTokens_[1], muonHandle);

          if (!muonHandle.failedToGet()) {
            for (reco::MuonCollection::const_iterator iMuon = muonHandle->begin(); iMuon != muonHandle->end();
                 ++iMuon) {
              if (!iMuon->isGlobalMuon())
                continue;

              reco::TrackRef checkedTrack = iMuon->outerTrack();
              if (muon.isTimeValid()) {
                // from bottom up
                if (checkedTrack->phi() < 0 && isUp) {
                  if (iMuon->time().timeAtIpInOut < corrTimeNeg_)
                    result = 1.0;
                  break;
                } else if (checkedTrack->phi() > 0 && !isUp) {
                  // from top down
                  if (iMuon->time().timeAtIpInOut < corrTimeNeg_)
                    result = 1.0;
                  break;
                }
              }  //muon is time valid
            }
          }
        }  //track is nonnull
      }    //double check timing
    } else {
      //case of a single muon event
      if (muon.time().timeAtIpInOut < offTimeNeg || muon.time().timeAtIpInOut > offTimePos)
        result = 1.;
    }
  }  //is time valid

  if (!isLoose && result > 0) {
    //check loose ip
    if (pvMatches(iEvent, muon, true) == 0)
      result *= 2.;
  }

  return result;
}

//
//Back-to-back selector
//
unsigned int MuonCosmicCompatibilityFiller::backToBack2LegCosmic(const edm::Event& iEvent,
                                                                 const reco::Muon& muon) const {
  unsigned int result = 0;  //no partners - collision
  reco::TrackRef track;
  if (muon.isGlobalMuon())
    track = muon.innerTrack();
  else if (muon.isTrackerMuon())
    track = muon.track();
  else if (muon.isStandAloneMuon() || muon.isRPCMuon() || muon.isGEMMuon() || muon.isME0Muon())
    return false;

  for (unsigned int iColl = 0; iColl < trackTokens_.size(); ++iColl) {
    edm::Handle<reco::TrackCollection> trackHandle;
    iEvent.getByToken(trackTokens_[iColl], trackHandle);
    if (muonid::findOppositeTrack(trackHandle, *track, angleThreshold_, deltaPt_).isNonnull()) {
      result++;
    }
  }  //loop over track collections

  return result;
}

//
//Check the number of global muons in an event, return true if there are more than 1 muon
//
unsigned int MuonCosmicCompatibilityFiller::nMuons(const edm::Event& iEvent) const {
  unsigned int nGlb = 0;

  edm::Handle<reco::MuonCollection> muonHandle;
  iEvent.getByToken(muonTokens_[1], muonHandle);

  if (!muonHandle.failedToGet()) {
    for (reco::MuonCollection::const_iterator iMuon = muonHandle->begin(); iMuon != muonHandle->end(); ++iMuon) {
      if (!iMuon->isGlobalMuon())
        continue;
      nGlb++;
    }
  }

  return nGlb;
}

//
//Check overlap between collections, use shared hits info
//
bool MuonCosmicCompatibilityFiller::isOverlappingMuon(const edm::Event& iEvent,
                                                      const edm::EventSetup& iSetup,
                                                      const reco::Muon& muon) const {
  // 4 steps in this module
  // step1 : check whether it's 1leg cosmic muon or not
  // step2 : both muons (muons and muonsFromCosmics1Leg) should have close IP
  // step3 : both muons should share very close reference point
  // step4 : check shared hits in both muon tracks

  // check if this muon is available in muonsFromCosmics collection
  bool overlappingMuon = false;  //false - not cosmic-like
  if (!muon.isGlobalMuon())
    return false;

  // reco muons for cosmics
  edm::Handle<reco::MuonCollection> muonHandle;
  iEvent.getByToken(cosmicToken_, muonHandle);

  // Global Tracking Geometry
  ESHandle<GlobalTrackingGeometry> trackingGeometry = iSetup.getHandle(geometryToken_);

  // PV
  math::XYZPoint RefVtx;
  RefVtx.SetXYZ(0, 0, 0);

  edm::Handle<reco::VertexCollection> pvHandle;
  iEvent.getByToken(vertexToken_, pvHandle);
  const reco::VertexCollection& vertices = *pvHandle.product();
  for (reco::VertexCollection::const_iterator it = vertices.begin(); it != vertices.end(); ++it) {
    RefVtx = it->position();
  }

  if (!muonHandle.failedToGet()) {
    for (reco::MuonCollection::const_iterator cosmicMuon = muonHandle->begin(); cosmicMuon != muonHandle->end();
         ++cosmicMuon) {
      if (cosmicMuon->innerTrack() == muon.innerTrack() || cosmicMuon->outerTrack() == muon.outerTrack())
        return true;

      reco::TrackRef outertrack = muon.outerTrack();
      reco::TrackRef costrack = cosmicMuon->outerTrack();

      // shared hits
      int RecHitsMuon = outertrack->numberOfValidHits();
      int shared = 0;
      // count hits for same hemisphere
      if (costrack.isNonnull()) {
        // unused
        //	bool isCosmic1Leg = false;
        //	bool isCloseIP = false;
        //	bool isCloseRef = false;

        for (trackingRecHit_iterator coshit = costrack->recHitsBegin(); coshit != costrack->recHitsEnd(); coshit++) {
          if ((*coshit)->isValid()) {
            DetId id((*coshit)->geographicalId());
          }
        }
        // step1
        //if( !isCosmic1Leg ) continue;

        if (outertrack.isNonnull()) {
          // step2
          //UNUSED:          const double ipErr = (double)outertrack->d0Error();
          //UNUSED:          double ipThreshold  = max(ipThreshold_, ipErr);
          //UNUSED:	  if( fabs(outertrack->dxy(RefVtx) + costrack->dxy(RefVtx)) < ipThreshold ) isCloseIP = true;
          //if( !isCloseIP ) continue;

          // step3
          GlobalPoint muonRefVtx(outertrack->vx(), outertrack->vy(), outertrack->vz());
          GlobalPoint cosmicRefVtx(costrack->vx(), costrack->vy(), costrack->vz());
          //UNUSED:	  float dist = (muonRefVtx - cosmicRefVtx).mag();
          //UNUSED:	  if( dist < 0.1 ) isCloseRef = true;
          //if( !isCloseRef ) continue;

          for (trackingRecHit_iterator trkhit = outertrack->recHitsBegin(); trkhit != outertrack->recHitsEnd();
               trkhit++) {
            if ((*trkhit)->isValid()) {
              for (trackingRecHit_iterator coshit = costrack->recHitsBegin(); coshit != costrack->recHitsEnd();
                   coshit++) {
                if ((*coshit)->isValid()) {
                  if ((*trkhit)->geographicalId() == (*coshit)->geographicalId()) {
                    if (((*trkhit)->localPosition() - (*coshit)->localPosition()).mag() < 10e-5)
                      shared++;
                  }
                }
              }
            }
          }
        }
      }
      // step4
      double fraction = -1;
      if (RecHitsMuon != 0)
        fraction = shared / (double)RecHitsMuon;
      if (shared > sharedHits_ && fraction > sharedFrac_) {
        overlappingMuon = true;
        break;
      }
    }
  }

  return overlappingMuon;
}

//
//pv matches
//
unsigned int MuonCosmicCompatibilityFiller::pvMatches(const edm::Event& iEvent,
                                                      const reco::Muon& muon,
                                                      bool isLoose) const {
  float maxdxyMult, maxdzMult, maxdxy, maxdz;

  if (isLoose) {
    //use "loose" parameter set
    maxdxyMult = maxdxyLooseMult_;
    maxdzMult = maxdzLooseMult_;
    maxdxy = maxdxyLoose_;
    maxdz = maxdzLoose_;
  } else {
    maxdxyMult = maxdxyTightMult_;
    maxdzMult = maxdzTightMult_;
    maxdxy = maxdxyTight_;
    maxdz = maxdzTight_;
  }

  unsigned int result = 0;

  reco::TrackRef track;
  if (muon.isGlobalMuon())
    track = muon.innerTrack();
  else if (muon.isTrackerMuon() || muon.isRPCMuon())
    track = muon.track();
  else if (muon.isStandAloneMuon())
    track = muon.standAloneMuon();

  bool multipleMu = false;
  if (nMuons(iEvent) > 1)
    multipleMu = true;

  math::XYZPoint RefVtx;
  RefVtx.SetXYZ(0, 0, 0);

  edm::Handle<reco::VertexCollection> pvHandle;
  iEvent.getByToken(vertexToken_, pvHandle);
  const reco::VertexCollection& vertices = *pvHandle.product();
  for (reco::VertexCollection::const_iterator it = vertices.begin(); it != vertices.end(); ++it) {
    RefVtx = it->position();

    if (track.isNonnull()) {
      if (multipleMu) {
        //multiple muon event

        if (fabs((*track).dxy(RefVtx)) < maxdxyMult || fabs((*track).dz(RefVtx)) < maxdzMult) {
          result++;

          //case of extra large dxy
          if (!isLoose && fabs((*track).dxy(RefVtx)) > largedxyMult_)
            result -= 1;
        }
      } else {
        //single muon event

        if (fabs((*track).dxy(RefVtx)) < maxdxy || fabs((*track).dz(RefVtx)) < maxdz) {
          result++;

          //case of extra large dxy
          if (!isLoose && fabs((*track).dxy(RefVtx)) > largedxy_)
            result -= 1;
        }
      }
    }  //track is nonnull
  }    //loop over vertices

  //special case for non-cosmic large ip muons
  if (result == 0 && multipleMu) {
    // consider all reco muons in an event
    edm::Handle<reco::MuonCollection> muonHandle;
    iEvent.getByToken(muonTokens_[1], muonHandle);

    //cosmic event should have zero good vertices
    edm::Handle<reco::VertexCollection> pvHandle;
    iEvent.getByToken(vertexToken_, pvHandle);
    const reco::VertexCollection& vertices = *pvHandle.product();

    //find the "other" one
    if (!muonHandle.failedToGet()) {
      for (reco::MuonCollection::const_iterator muons = muonHandle->begin(); muons != muonHandle->end(); ++muons) {
        if (!muons->isGlobalMuon())
          continue;
        //skip this track
        if (muons->innerTrack() == muon.innerTrack() && muons->outerTrack() == muon.outerTrack())
          continue;
        //check ip and vertex of the "other" muon
        reco::TrackRef tracks;
        if (muons->isGlobalMuon())
          tracks = muons->innerTrack();
        if (fabs((*tracks).dxy(RefVtx)) > hIpTrdxy_)
          continue;
        //check if vertex collection is empty
        if (vertices.begin() == vertices.end())
          continue;
        //for(reco::VertexCollection::const_iterator it=vertices.begin() ; it!=vertices.end() ; ++it) {
        //find matching vertex by position
        //if (fabs(it->z() - tracks->vz()) > 0.01) continue; //means will not be untagged from cosmics
        if (TMath::Prob(vertices.front().chi2(), (int)(vertices.front().ndof())) > hIpTrvProb_)
          result = 1;
        //}
      }
    }
  }

  return result;
}

float MuonCosmicCompatibilityFiller::combinedCosmicID(const edm::Event& iEvent,
                                                      const edm::EventSetup& iSetup,
                                                      const reco::Muon& muon,
                                                      bool CheckMuonID,
                                                      bool checkVertex) const {
  float result = 0.0;

  // return >=1 = identify as cosmic muon (the more like cosmics, the higher is the number)
  // return 0.0 = identify as collision muon
  if (muon.isGlobalMuon()) {
    unsigned int cosmicVertex = eventActivity(iEvent, muon);
    bool isOverlapping = isOverlappingMuon(iEvent, iSetup, muon);
    unsigned int looseIp = pvMatches(iEvent, muon, true);
    unsigned int tightIp = pvMatches(iEvent, muon, false);
    float looseTime = muonTiming(iEvent, muon, true);
    float tightTime = muonTiming(iEvent, muon, false);
    unsigned int backToback = backToBack2LegCosmic(iEvent, muon);
    //bool cosmicSegment = checkMuonSegments(muon);

    //short cut to reject cosmic event
    if (checkVertex && cosmicVertex == 0)
      return 10.0;

    // compatibility (0 - 10)
    // weight is assigned by the performance of individual module
    // btob: ~90% eff / ~0% misid
    // ip: ~90% eff / ~0% misid
    // time: ~30% eff / ~0% misid
    double weight_btob = 2.0;
    double weight_ip = 2.0;
    double weight_time = 1.0;
    double weight_overlap = 0.5;

    // collision muon should have compatibility < 4 (0 - 4)
    // cosmic muon should have compatibility >= 4 (4 - 10)

    // b-to-b (max comp.: 4.0)
    if (backToback >= 1) {
      //in this case it is cosmic for sure
      result += weight_btob * 2.;
      if (tightIp == 1) {
        // check with other observables to reduce mis-id (subtract compatibilities)
        if (looseIp == 1) {
          if (backToback < 2)
            result -= weight_btob * 0.5;
        }
      }
    }

    // ip (max comp.: 4.0)
    if (tightIp == 0) {
      //in this case it is cosmic for sure
      result += weight_ip * 2.0;
      if (backToback == 0) {
        // check with other observables to reduce mis-id (subtract compatibilities)
        if (tightTime == 0) {
          if (looseTime == 0 && !isOverlapping)
            result -= weight_ip * 1.0;
        }
      }
    } else if (tightIp >= 2) {
      // in this case it is almost collision-like (reduce compatibility)
      // if multi pvs: comp = -2.0
      if (backToback >= 1)
        result -= weight_ip * 1.0;
    }

    // timing (max comp.: 2.0)
    if (tightTime > 0) {
      // bonus track
      if (looseTime > 0) {
        if (backToback >= 1) {
          if (tightIp == 0)
            result += weight_time * tightTime;
          else if (looseIp == 0)
            result += weight_time * 0.25;
        }
      } else {
        if (backToback >= 1 && tightIp == 0)
          result += weight_time * 0.25;
      }
    }

    // overlapping
    if (backToback == 0 && isOverlapping) {
      // bonus track
      if (tightIp == 0 && tightTime >= 1) {
        result += weight_overlap * 1.0;
      }
    }
  }  //is global muon

  //  if (CheckMuonID && cosmicSegment) result += 4;

  return result;
}

//
//Track activity/vertex quality, count good vertices
//
unsigned int MuonCosmicCompatibilityFiller::eventActivity(const edm::Event& iEvent, const reco::Muon& muon) const {
  unsigned int result = 0;  //no good vertices - cosmic-like

  //check track activity
  edm::Handle<reco::TrackCollection> tracks;
  iEvent.getByToken(trackTokens_[0], tracks);
  if (!tracks.failedToGet() && tracks->size() < 3)
    return 0;

  //cosmic event should have zero good vertices
  edm::Handle<reco::VertexCollection> pvHandle;
  if (!iEvent.getByToken(vertexToken_, pvHandle)) {
    return 0;
  } else {
    const reco::VertexCollection& vertices = *pvHandle.product();
    //check if vertex collection is empty
    if (vertices.begin() == vertices.end())
      return 0;
    for (reco::VertexCollection::const_iterator it = vertices.begin(); it != vertices.end(); ++it) {
      if ((TMath::Prob(it->chi2(), (int)it->ndof()) > minvProb_) && (fabs(it->z()) <= maxvertZ_) &&
          (fabs(it->position().rho()) <= maxvertRho_))
        result++;
    }
  }
  return result;
}

//
//Muon iD variables
//
bool MuonCosmicCompatibilityFiller::checkMuonID(const reco::Muon& imuon) const {
  bool result = false;
  // initial set up using Jordan's study: GlobalMuonPromptTight + TMOneStationLoose
  if (muon::isGoodMuon(imuon, muon::GlobalMuonPromptTight) && muon::isGoodMuon(imuon, muon::TMOneStationLoose))
    result = true;

  return result;
}

bool MuonCosmicCompatibilityFiller::checkMuonSegments(const reco::Muon& imuon) const {
  bool result = false;
  if (imuon.numberOfMatches() < nChamberMatches_ && muon::segmentCompatibility(imuon) < segmentComp_)
    result = true;

  return result;
}
