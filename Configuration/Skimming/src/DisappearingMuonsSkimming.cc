// -*- C++ -*-
//
// Package:    MuMu/DisappearingMuonsSkimming
// Class:      DisappearingMuonsSkimming
//
/**\class DisappearingMuonsSkimming DisappearingMuonsSkimming_AOD.cc MuMu/DisappearingMuonsSkimming/plugins/DisappearingMuonsSkimming_AOD.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Michael Revering
//         Created:  Mon, 18 Jun 2018 21:22:23 GMT
//
//
// system include files
#include <memory>
#include "Math/VectorUtil.h"

// user include files
#include "Configuration/Skimming/interface/DisappearingMuonsSkimming.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

DisappearingMuonsSkimming::DisappearingMuonsSkimming(const edm::ParameterSet& iConfig)
    : recoMuonToken_(consumes<reco::MuonCollection>(iConfig.getParameter<edm::InputTag>("recoMuons"))),
      standaloneMuonToken_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("StandaloneTracks"))),
      trackCollectionToken_(consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracks"))),
      primaryVerticesToken_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("primaryVertices"))),
      reducedEndcapRecHitCollectionToken_(
          consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("EERecHits"))),
      reducedBarrelRecHitCollectionToken_(
          consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("EBRecHits"))),
      trigResultsToken_(consumes<edm::TriggerResults>(iConfig.getParameter<edm::InputTag>("TriggerResultsTag"))),
      transientTrackToken_(
          esConsumes<TransientTrackBuilder, TransientTrackRecord>(edm::ESInputTag("", "TransientTrackBuilder"))),
      geometryToken_(esConsumes<CaloGeometry, CaloGeometryRecord>(edm::ESInputTag{})),
      muonPathsToPass_(iConfig.getParameter<std::vector<std::string>>("muonPathsToPass")),
      minMuPt_(iConfig.getParameter<double>("minMuPt")),
      maxMuEta_(iConfig.getParameter<double>("maxMuEta")),
      minTrackEta_(iConfig.getParameter<double>("minTrackEta")),
      maxTrackEta_(iConfig.getParameter<double>("maxTrackEta")),
      minTrackPt_(iConfig.getParameter<double>("minTrackPt")),
      maxTransDCA_(iConfig.getParameter<double>("maxTransDCA")),
      maxLongDCA_(iConfig.getParameter<double>("maxLongDCA")),
      maxVtxChi_(iConfig.getParameter<double>("maxVtxChi")),
      minInvMass_(iConfig.getParameter<double>("minInvMass")),
      maxInvMass_(iConfig.getParameter<double>("maxInvMass")),
      trackIsoConesize_(iConfig.getParameter<double>("trackIsoConesize")),
      trackIsoInnerCone_(iConfig.getParameter<double>("trackIsoInnerCone")),
      ecalIsoConesize_(iConfig.getParameter<double>("ecalIsoConesize")),
      minEcalHitE_(iConfig.getParameter<double>("minEcalHitE")),
      maxTrackIso_(iConfig.getParameter<double>("maxTrackIso")),
      maxEcalIso_(iConfig.getParameter<double>("maxEcalIso")),
      minSigInvMass_(iConfig.getParameter<double>("minSigInvMass")),
      maxSigInvMass_(iConfig.getParameter<double>("maxSigInvMass")),
      minStandaloneDr_(iConfig.getParameter<double>("minStandaloneDr")),
      maxStandaloneDE_(iConfig.getParameter<double>("maxStandaloneDE")),
      keepOffPeak_(iConfig.getParameter<bool>("keepOffPeak")),
      keepSameSign_(iConfig.getParameter<bool>("keepSameSign")),
      keepTotalRegion_(iConfig.getParameter<bool>("keepTotalRegion")),
      keepPartialRegion_(iConfig.getParameter<bool>("keepPartialRegion")) {}

//
// member functions
//

// ------------ method called for each event  ------------
bool DisappearingMuonsSkimming::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;
  using namespace reco;

  bool totalRegion = false;
  bool sameSign = false;
  bool offPeak = false;
  bool partialRegion = false;

  const auto& staTracks = iEvent.get(standaloneMuonToken_);
  const auto& vertices = iEvent.get(primaryVerticesToken_);
  const auto& recoMuons = iEvent.get(recoMuonToken_);
  const auto& thePATTracks = iEvent.get(trackCollectionToken_);
  const auto& triggerResults = iEvent.get(trigResultsToken_);

  // this wraps tracks with additional methods that are used in vertex-calculation
  const TransientTrackBuilder* transientTrackBuilder = &iSetup.getData(transientTrackToken_);

  if (!passTriggers(iEvent, triggerResults, muonPathsToPass_))
    return false;

  int nMuonTrackCand = 0;
  float MuonTrackMass = 0.;

  //Looping over the reconstructed Muons
  for (const auto& iMuon : recoMuons) {
    if (!(iMuon.isPFMuon() && iMuon.isGlobalMuon()))
      continue;
    if (!(iMuon.passed(reco::Muon::CutBasedIdTight)))
      continue;
    if (!(iMuon.passed(reco::Muon::PFIsoTight)))
      continue;
    if (iMuon.pt() < minMuPt_ || std::abs(iMuon.eta()) > maxMuEta_)
      continue;

    //Looping over tracks for any good muon
    int indx(0);
    for (const auto& iTrack : thePATTracks) {
      reco::TrackRef trackRef = reco::TrackRef(&thePATTracks, indx);
      indx++;

      if (!iTrack.quality(reco::Track::qualityByName("highPurity")))
        continue;
      if (std::abs(iTrack.eta()) > maxTrackEta_ || std::abs(iTrack.eta()) < minTrackEta_)
        continue;
      if (iTrack.pt() < minTrackPt_)
        continue;
      //Check if the track belongs to a primary vertex for isolation calculation

      unsigned int vtxIndex;
      unsigned int tkIndex;
      bool foundtrack = this->findTrackInVertices(trackRef, vertices, vtxIndex, tkIndex);

      if (!foundtrack)
        continue;

      reco::VertexRef vtx(&vertices, vtxIndex);
      GlobalPoint tkVtx = GlobalPoint(vtx->x(), vtx->y(), vtx->z());

      reco::TransientTrack tk = transientTrackBuilder->build(iTrack);
      TrajectoryStateClosestToPoint traj = tk.trajectoryStateClosestToPoint(tkVtx);
      double transDCA = traj.perigeeParameters().transverseImpactParameter();
      double longDCA = traj.perigeeParameters().longitudinalImpactParameter();
      if (std::abs(longDCA) > maxLongDCA_)
        continue;
      if (std::abs(transDCA) > maxTransDCA_)
        continue;
      // make a pair of TransientTracks to feed the vertexer
      std::vector<reco::TransientTrack> tracksToVertex;
      tracksToVertex.push_back(tk);
      tracksToVertex.push_back(transientTrackBuilder->build(iMuon.globalTrack()));
      // try to fit these two tracks to a common vertex
      KalmanVertexFitter vertexFitter;
      CachingVertex<5> fittedVertex = vertexFitter.vertex(tracksToVertex);
      double vtxChi = 0;
      // some poor fits will simply fail to find a common vertex
      if (fittedVertex.isValid() && fittedVertex.totalChiSquared() >= 0. && fittedVertex.degreesOfFreedom() > 0) {
        // others we can exclude by their poor fit
        vtxChi = fittedVertex.totalChiSquared() / fittedVertex.degreesOfFreedom();

        if (vtxChi < maxVtxChi_) {
          // important! evaluate momentum vectors AT THE VERTEX
          TrajectoryStateClosestToPoint one_TSCP =
              tracksToVertex[0].trajectoryStateClosestToPoint(fittedVertex.position());
          TrajectoryStateClosestToPoint two_TSCP =
              tracksToVertex[1].trajectoryStateClosestToPoint(fittedVertex.position());
          GlobalVector one_momentum = one_TSCP.momentum();
          GlobalVector two_momentum = two_TSCP.momentum();

          double total_energy = sqrt(one_momentum.mag2() + 0.106 * 0.106) + sqrt(two_momentum.mag2() + 0.106 * 0.106);
          double total_px = one_momentum.x() + two_momentum.x();
          double total_py = one_momentum.y() + two_momentum.y();
          double total_pz = one_momentum.z() + two_momentum.z();
          MuonTrackMass = sqrt(pow(total_energy, 2) - pow(total_px, 2) - pow(total_py, 2) - pow(total_pz, 2));
        } else {
          continue;
        }
      } else {
        continue;
      }
      if (MuonTrackMass < minInvMass_ || MuonTrackMass > maxInvMass_)
        continue;

      double trackIso = getTrackIsolation(trackRef, vertices);
      //Track iso returns -1 when it fails to find the vertex containing the track (already checked in track selection, but might as well make sure)
      if (trackIso < 0)
        continue;
      double ecalIso = getECALIsolation(iEvent, iSetup, transientTrackBuilder->build(iTrack));
      if (trackIso > maxTrackIso_ || ecalIso > maxEcalIso_)
        continue;

      //A good tag/probe pair has been selected, now check for control or signal regions
      if (iMuon.charge() == iTrack.charge()) {
        sameSign = true;
      }

      //If not same sign CR, need to check standalone muons for signal regions
      double staMinDr2 = 1000;
      double staMinDEoverE = -10;
      if (!staTracks.empty()) {
        for (const auto& staTrack : staTracks) {
          reco::TransientTrack track = transientTrackBuilder->build(staTrack);
          double dR2 = deltaR2(track.impactPointTSCP().momentum().eta(),
                               track.impactPointTSCP().momentum().phi(),
                               iTrack.eta(),
                               iTrack.phi());
          double staDE = (std::sqrt(track.impactPointTSCP().momentum().mag2()) - iTrack.p()) / iTrack.p();
          if (dR2 < staMinDr2) {
            staMinDr2 = dR2;
          }
          //Pick the largest standalone muon within the cone
          if (dR2 < minStandaloneDr_ * minStandaloneDr_ && staDE > staMinDEoverE) {
            staMinDEoverE = staDE;
          }
        }
      }
      if (staMinDr2 > minStandaloneDr_ * minStandaloneDr_) {
        if (MuonTrackMass < minSigInvMass_ || MuonTrackMass > maxSigInvMass_) {
          offPeak = true;
        } else {
          totalRegion = true;
        }
      } else {
        if (staMinDEoverE < maxStandaloneDE_) {
          if (MuonTrackMass < minSigInvMass_ || MuonTrackMass > maxSigInvMass_) {
            offPeak = true;
          } else {
            partialRegion = true;
          }
        }
      }
      nMuonTrackCand++;
    }
  }

  if (nMuonTrackCand == 0)
    return false;
  bool passes = false;
  //Pass all same sign CR events
  if (sameSign && keepSameSign_) {
    passes = true;
  }
  //Pass all total disappearance events
  else if (totalRegion && keepTotalRegion_) {
    passes = true;
  }
  //Pass all partial disappearkance off-peak events
  else if (offPeak && keepOffPeak_) {
    passes = true;
  }
  //Pass partial region events that pass minimum standalone muon DE/E.
  else if (partialRegion && keepPartialRegion_) {
    passes = true;
  }
  return passes;
}

bool DisappearingMuonsSkimming::passTriggers(const edm::Event& iEvent,
                                             const edm::TriggerResults& triggerResults,
                                             std::vector<std::string> m_muonPathsToPass) {
  bool passTriggers = false;
  const edm::TriggerNames& trigNames = iEvent.triggerNames(triggerResults);
  for (size_t i = 0; i < trigNames.size(); ++i) {
    const std::string& name = trigNames.triggerName(i);
    for (auto& pathName : m_muonPathsToPass) {
      if ((name.find(pathName) != std::string::npos)) {
        if (triggerResults.accept(i)) {
          passTriggers = true;
          break;
        }
      }
    }
  }
  return passTriggers;
}

bool DisappearingMuonsSkimming::findTrackInVertices(const reco::TrackRef& tkToMatch,
                                                    const reco::VertexCollection& vertices,
                                                    unsigned int& vtxIndex,
                                                    unsigned int& trackIndex) {
  // initialize indices
  vtxIndex = -1;
  trackIndex = -1;

  bool foundtrack{false};
  unsigned int idx = 0;
  for (const auto& vtx : vertices) {
    if (!vtx.isValid()) {
      idx++;
      continue;
    }

    const auto& thePVtracks{vtx.tracks()};
    std::vector<unsigned int> thePVkeys;
    thePVkeys.reserve(thePVtracks.size());
    for (const auto& tv : thePVtracks) {
      thePVkeys.push_back(tv.key());
    }

    auto result = std::find_if(
        thePVkeys.begin(), thePVkeys.end(), [tkToMatch](const auto& tkRef) { return tkRef == tkToMatch.key(); });
    if (result != thePVkeys.end()) {
      foundtrack = true;
      trackIndex = std::distance(thePVkeys.begin(), result);
      vtxIndex = idx;
    }
    if (foundtrack)
      break;

    idx++;
  }
  return foundtrack;
}

double DisappearingMuonsSkimming::getTrackIsolation(const reco::TrackRef& tkToMatch,
                                                    const reco::VertexCollection& vertices) {
  unsigned int vtxindex;
  unsigned int trackIndex;
  bool foundtrack = this->findTrackInVertices(tkToMatch, vertices, vtxindex, trackIndex);

  LogDebug("DisappearingMuonsSkimming") << "getTrackIsolation vtx Index: " << vtxindex
                                        << " track Index: " << trackIndex;

  if (!foundtrack) {
    return -1.;
  }

  reco::VertexRef primaryVtx(&vertices, vtxindex);

  double Isolation = 0;
  for (unsigned int i = 0; i < primaryVtx->tracksSize(); i++) {
    if (i == trackIndex)
      continue;
    reco::TrackBaseRef secondarytrack = primaryVtx->trackRefAt(i);
    if (deltaR2(tkToMatch.get()->eta(), tkToMatch.get()->phi(), secondarytrack->eta(), secondarytrack->phi()) >
            trackIsoConesize_ * trackIsoConesize_ ||
        deltaR2(tkToMatch.get()->eta(), tkToMatch.get()->phi(), secondarytrack->eta(), secondarytrack->phi()) <
            trackIsoInnerCone_ * trackIsoInnerCone_)
      continue;
    Isolation += secondarytrack->pt();
  }

  return Isolation / tkToMatch.get()->pt();
}

double DisappearingMuonsSkimming::getECALIsolation(const edm::Event& iEvent,
                                                   const edm::EventSetup& iSetup,
                                                   const reco::TransientTrack track) {
  edm::Handle<EcalRecHitCollection> rechitsEE;
  iEvent.getByToken(reducedEndcapRecHitCollectionToken_, rechitsEE);

  edm::Handle<EcalRecHitCollection> rechitsEB;
  iEvent.getByToken(reducedBarrelRecHitCollectionToken_, rechitsEB);

  const CaloGeometry& caloGeom = iSetup.getData(geometryToken_);
  TrajectoryStateClosestToPoint t0 = track.impactPointTSCP();
  double eDR = 0;

  for (EcalRecHitCollection::const_iterator hit = rechitsEE->begin(); hit != rechitsEE->end(); hit++) {
    const DetId id = (*hit).detid();
    const GlobalPoint hitPos = caloGeom.getSubdetectorGeometry(id)->getGeometry(id)->getPosition();
    //Check if hit and track trajectory ar in the same endcap (transient track projects both ways)
    if ((hitPos.eta() * t0.momentum().eta()) < 0) {
      continue;
    }
    TrajectoryStateClosestToPoint traj = track.trajectoryStateClosestToPoint(hitPos);
    math::XYZVector idPositionRoot(hitPos.x(), hitPos.y(), hitPos.z());
    math::XYZVector trajRoot(traj.position().x(), traj.position().y(), traj.position().z());
    if (deltaR2(idPositionRoot, trajRoot) < ecalIsoConesize_ * ecalIsoConesize_ && (*hit).energy() > minEcalHitE_) {
      eDR += (*hit).energy();
    }
  }
  for (EcalRecHitCollection::const_iterator hit = rechitsEB->begin(); hit != rechitsEB->end(); hit++) {
    const DetId id = (*hit).detid();
    const GlobalPoint hitPos = caloGeom.getSubdetectorGeometry(id)->getGeometry(id)->getPosition();
    if ((hitPos.eta() * t0.momentum().eta()) < 0) {
      continue;
    }
    TrajectoryStateClosestToPoint traj = track.trajectoryStateClosestToPoint(hitPos);
    math::XYZVector idPositionRoot(hitPos.x(), hitPos.y(), hitPos.z());
    math::XYZVector trajRoot(traj.position().x(), traj.position().y(), traj.position().z());
    if (deltaR2(idPositionRoot, trajRoot) < ecalIsoConesize_ * ecalIsoConesize_ && (*hit).energy() > minEcalHitE_) {
      eDR += (*hit).energy();
    }
  }

  return eDR;
}

void DisappearingMuonsSkimming::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("recoMuons", edm::InputTag("muons"));
  desc.add<edm::InputTag>("tracks", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("StandaloneTracks", edm::InputTag("standAloneMuons"));
  desc.add<edm::InputTag>("primaryVertices", edm::InputTag("offlinePrimaryVertices"));
  desc.add<edm::InputTag>("EERecHits", edm::InputTag("reducedEcalRecHitsEE"));
  desc.add<edm::InputTag>("EBRecHits", edm::InputTag("reducedEcalRecHitsEB"));
  desc.add<edm::InputTag>("TriggerResultsTag", edm::InputTag("TriggerResults", "", "HLT"));
  desc.add<std::vector<std::string>>("muonPathsToPass",
                                     {
                                         "HLT_IsoMu24_v",
                                         "HLT_IsoMu27_v",
                                     });
  desc.add<double>("minMuPt", 26);
  desc.add<double>("maxMuEta", 2.4);
  desc.add<double>("minTrackEta", 0);
  desc.add<double>("maxTrackEta", 2.4);
  desc.add<double>("minTrackPt", 20);
  desc.add<double>("maxTransDCA", 0.005);
  desc.add<double>("maxLongDCA", 0.05);
  desc.add<double>("maxVtxChi", 3.0);
  desc.add<double>("minInvMass", 50);
  desc.add<double>("maxInvMass", 150);
  desc.add<double>("trackIsoConesize", 0.3);
  desc.add<double>("trackIsoInnerCone", 0.01);
  desc.add<double>("ecalIsoConesize", 0.4);
  desc.add<double>("minEcalHitE", 0.3);
  desc.add<double>("maxTrackIso", 0.05);
  desc.add<double>("maxEcalIso", 10);
  desc.add<double>("minSigInvMass", 76);
  desc.add<double>("maxSigInvMass", 106);
  desc.add<double>("minStandaloneDr", 1.0);
  desc.add<double>("maxStandaloneDE", -0.5);
  desc.add<bool>("keepOffPeak", true);
  desc.add<bool>("keepSameSign", true);
  desc.add<bool>("keepTotalRegion", true);
  desc.add<bool>("keepPartialRegion", true);
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(DisappearingMuonsSkimming);
