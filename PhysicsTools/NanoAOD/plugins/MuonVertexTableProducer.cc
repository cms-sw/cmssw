#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/transform.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "TLorentzVector.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/PatCandidates/interface/Muon.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistance.h"
#include "PhysicsTools/RecoUtils/interface/CheckHitPattern.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "TrackingTools/IPTools/interface/IPTools.h"

#include <vector>
#include <iostream>

class MuonVertexTableProducer : public edm::global::EDProducer<> {
public:
  explicit MuonVertexTableProducer(const edm::ParameterSet& iConfig)
      : dsaMuonTag_(consumes<std::vector<reco::Track>>(iConfig.getParameter<edm::InputTag>("dsaMuons"))),
        patMuonTag_(consumes<std::vector<pat::Muon>>(iConfig.getParameter<edm::InputTag>("patMuons"))),
        bsTag_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamspot"))),
        pvTag_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("primaryVertex"))),
        generalTrackTag_(consumes<std::vector<reco::Track>>(iConfig.getParameter<edm::InputTag>("generalTracks"))),
        propagatorToken_(esConsumes(edm::ESInputTag("", "SteppingHelixPropagatorAny"))),
        magneticFieldToken_(esConsumes<MagneticField, IdealMagneticFieldRecord>()),
        tkerGeomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>()),
        tkerTopoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd>()),
        transientTrackBuilderToken_(esConsumes(edm::ESInputTag("", "TransientTrackBuilder"))) {
    produces<nanoaod::FlatTable>("PatMuonVertex");
    produces<nanoaod::FlatTable>("PatDSAMuonVertex");
    produces<nanoaod::FlatTable>("DSAMuonVertex");
    produces<nanoaod::FlatTable>("PatMuonVertexRefittedTracks");
    produces<nanoaod::FlatTable>("PatDSAMuonVertexRefittedTracks");
    produces<nanoaod::FlatTable>("DSAMuonVertexRefittedTracks");
  }

  ~MuonVertexTableProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("patMuons")->setComment("input pat muon collection");
    desc.add<edm::InputTag>("dsaMuons")->setComment("input displaced standalone muon collection");
    desc.add<edm::InputTag>("beamspot")->setComment("input beamspot collection");
    desc.add<edm::InputTag>("primaryVertex")->setComment("input primaryVertex collection");
    desc.add<edm::InputTag>("generalTracks")->setComment("input generalTracks collection");
    descriptions.add("muonVertexTables", desc);
  }

private:
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

  std::pair<float, float> getVxy(const reco::Vertex muonVertex) const;
  std::pair<float, float> getVxyz(const reco::Vertex muonVertex) const;

  template <typename MuonType1 = pat::Muon, typename MuonType2>
  float getDisplacedTrackerIsolation(const std::vector<reco::Track>& generalTracks,
                                     const MuonType1& muon_1,
                                     const reco::Vertex muonVertex,
                                     const reco::BeamSpot& beamspot,
                                     const MuonType2* muon_2 = nullptr,
                                     float maxDR = 0.3,
                                     float minDR = 0.01,
                                     float maxDz = 0.5,
                                     float maxDxy = 0.2) const;

  template <typename MuonType = reco::Track>
  float getProximityDeltaR(const MuonType& track,
                           const MuonType& trackRef,
                           const edm::ESHandle<MagneticField>& magneticField,
                           const edm::ESHandle<Propagator>& propagator) const;

  std::tuple<float, float, GlobalPoint> getDistanceBetweenMuonTracks(
      const reco::Track& track1, const reco::Track& track2, const edm::ESHandle<MagneticField>& magneticField) const;

  const edm::EDGetTokenT<std::vector<reco::Track>> dsaMuonTag_;
  const edm::EDGetTokenT<std::vector<pat::Muon>> patMuonTag_;
  const edm::EDGetTokenT<reco::BeamSpot> bsTag_;
  const edm::EDGetTokenT<reco::VertexCollection> pvTag_;
  const edm::EDGetTokenT<std::vector<reco::Track>> generalTrackTag_;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> propagatorToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magneticFieldToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkerGeomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tkerTopoToken_;
  const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> transientTrackBuilderToken_;
};

void MuonVertexTableProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  const std::vector<reco::Track>& dsaMuons = iEvent.get(dsaMuonTag_);
  const std::vector<pat::Muon>& patMuons = iEvent.get(patMuonTag_);

  const reco::BeamSpot& beamSpotInput = iEvent.get(bsTag_);
  const auto& bs = beamSpotInput.position();
  GlobalPoint beamSpot(bs.x(), bs.y(), bs.z());
  reco::Vertex beamSpotVertex(beamSpotInput.position(), beamSpotInput.covariance3D());

  const reco::VertexCollection& primaryVertices = iEvent.get(pvTag_);
  const auto& pv = primaryVertices.at(0);
  GlobalPoint primaryVertex(pv.x(), pv.y(), pv.z());

  const std::vector<reco::Track>& generalTracks = iEvent.get(generalTrackTag_);

  auto const& propagator = &iSetup.getData(propagatorToken_);
  auto const& magneticField = &iSetup.getData(magneticFieldToken_);

  auto const& tkerGeom = &iSetup.getData(tkerGeomToken_);
  auto const& tkerTopo = &iSetup.getData(tkerTopoToken_);
  const TransientTrackBuilder& builder = iSetup.getData(transientTrackBuilderToken_);

  KalmanVertexFitter vertexFitter(true, true);

  int nPatPatVertices = 0;
  int nPatDSAVertices = 0;
  int nDSADSAVertices = 0;
  int ppRefittedTrackIdx_counter = 0;
  int pdRefittedTrackIdx_counter = 0;
  int ddRefittedTrackIdx_counter = 0;

  std::map<std::string, std::vector<bool>> vertexIsValid;
  std::map<std::string, std::vector<float>> vxy, vxyz, vx, vy, vz, t, vxySigma, vxyzSigma, vxErr, vyErr, vzErr, tErr;
  std::map<std::string, std::vector<float>> chi2, ndof, normChi2, dR, originalMuonIdx1, originalMuonIdx2,
      refittedTrackIdx1, refittedTrackIdx2, isDSAMuon1, isDSAMuon2;
  std::map<std::string, std::vector<float>> displacedTrackIso03Dimuon1, displacedTrackIso03Dimuon2,
      displacedTrackIso04Dimuon1, displacedTrackIso04Dimuon2;
  std::map<std::string, std::vector<float>> displacedTrackIso03Muon1, displacedTrackIso03Muon2,
      displacedTrackIso04Muon1, displacedTrackIso04Muon2, proxDeltaR;
  std::map<std::string, std::vector<float>> DCA, DCAstatus, DCAx, DCAy, DCAz;
  std::map<std::string, std::vector<float>> hitsInFrontOfVert1, missHitsAfterVert1, hitsInFrontOfVert2,
      missHitsAfterVert2;

  std::map<std::string, std::vector<float>> refittedTrackIdx, refittedTrackIsDSAMuon, refittedTrackOriginalIdx,
      refittedTrackPt, refittedTrackPtErr, refittedTrackPx, refittedTrackPy, refittedTrackPz;
  std::map<std::string, std::vector<float>> refittedTrackEta, refittedTrackEtaErr, refittedTrackPhi,
      refittedTrackPhiErr, refittedTrackCharge, refittedTrackNormChi2, refittedTrackNdof, refittedTrackChi2;
  std::map<std::string, std::vector<float>> refittedTrackDzPV, refittedTrackDzPVErr, refittedTrackDxyPVTraj,
      refittedTrackDxyPVTrajErr, refittedTrackDxyPVSigned, refittedTrackDxyPVSignedErr, refittedTrackIp3DPVSigned,
      refittedTrackIp3DPVSignedErr;
  std::map<std::string, std::vector<float>> refittedTrackIso03Dimuon1, refittedTrackIso03Dimuon2,
      refittedTrackIso04Dimuon1, refittedTrackIso04Dimuon2, refittedTrackIso03Muon1, refittedTrackIso03Muon2,
      refittedTrackIso04Muon1, refittedTrackIso04Muon2;

  // pat muons
  for (size_t i = 0; i < patMuons.size(); i++) {
    const pat::Muon& muon_i = patMuons.at(i);

    reco::TrackRef trackRef_i;
    if (muon_i.isGlobalMuon())
      trackRef_i = muon_i.combinedMuon();
    else if (muon_i.isStandAloneMuon())
      trackRef_i = muon_i.standAloneMuon();
    else
      trackRef_i = muon_i.tunePMuonBestTrack();

    const auto& muonTrack_i = trackRef_i.get();
    reco::TransientTrack muonTransientTrack_i = builder.build(muonTrack_i);

    // pat-pat muon vertex
    for (size_t j = i + 1; j < patMuons.size(); j++) {
      const pat::Muon& muon_j = patMuons.at(j);

      reco::TrackRef trackRef_j;
      if (muon_j.isGlobalMuon())
        trackRef_j = muon_j.combinedMuon();
      else if (muon_j.isStandAloneMuon())
        trackRef_j = muon_j.standAloneMuon();
      else
        trackRef_j = muon_j.tunePMuonBestTrack();

      const auto& muonTrack_j = trackRef_j.get();
      reco::TransientTrack muonTransientTrack_j = builder.build(muonTrack_j);

      std::vector<reco::TransientTrack> muonTransientTracks{};
      muonTransientTracks.push_back(muonTransientTrack_i);
      muonTransientTracks.push_back(muonTransientTrack_j);

      TransientVertex transientMuonVertex = vertexFitter.vertex(muonTransientTracks);
      reco::Vertex muonVertex = reco::Vertex(transientMuonVertex);

      if (!transientMuonVertex.isValid())
        continue;
      std::tuple<float, float, GlobalPoint> distanceTuple =
          getDistanceBetweenMuonTracks(*muonTrack_i, *muonTrack_j, magneticField);
      // if dca status is good but dca is more than 15 cm
      if (std::get<1>(distanceTuple) && std::get<0>(distanceTuple) > 15)
        continue;

      nPatPatVertices++;

      vertexIsValid["PATPAT"].push_back(transientMuonVertex.isValid());
      std::pair<float, float> vxy_ = getVxy(muonVertex);
      vxy["PATPAT"].push_back(vxy_.first);
      vxySigma["PATPAT"].push_back(vxy_.second);
      std::pair<float, float> vxyz_ = getVxyz(muonVertex);
      vxyz["PATPAT"].push_back(vxyz_.first);
      vxyzSigma["PATPAT"].push_back(vxyz_.second);
      chi2["PATPAT"].push_back(muonVertex.chi2());
      ndof["PATPAT"].push_back(muonVertex.ndof());
      normChi2["PATPAT"].push_back(muonVertex.normalizedChi2());
      vx["PATPAT"].push_back(muonVertex.x());
      vy["PATPAT"].push_back(muonVertex.y());
      vz["PATPAT"].push_back(muonVertex.z());
      t["PATPAT"].push_back(muonVertex.t());
      vxErr["PATPAT"].push_back(muonVertex.xError());
      vyErr["PATPAT"].push_back(muonVertex.yError());
      vzErr["PATPAT"].push_back(muonVertex.zError());
      tErr["PATPAT"].push_back(muonVertex.tError());

      dR["PATPAT"].push_back(reco::deltaR(muon_i, muon_j));
      originalMuonIdx1["PATPAT"].push_back(i);
      originalMuonIdx2["PATPAT"].push_back(j);
      isDSAMuon1["PATPAT"].push_back(0);
      isDSAMuon2["PATPAT"].push_back(0);

      displacedTrackIso03Dimuon1["PATPAT"].push_back(getDisplacedTrackerIsolation<pat::Muon, pat::Muon>(
          generalTracks, muon_i, muonVertex, beamSpotInput, &muon_j, 0.3));
      displacedTrackIso04Dimuon1["PATPAT"].push_back(getDisplacedTrackerIsolation<pat::Muon, pat::Muon>(
          generalTracks, muon_i, muonVertex, beamSpotInput, &muon_j, 0.4));
      displacedTrackIso03Dimuon2["PATPAT"].push_back(getDisplacedTrackerIsolation<pat::Muon, pat::Muon>(
          generalTracks, muon_j, muonVertex, beamSpotInput, &muon_i, 0.3));
      displacedTrackIso04Dimuon2["PATPAT"].push_back(getDisplacedTrackerIsolation<pat::Muon, pat::Muon>(
          generalTracks, muon_j, muonVertex, beamSpotInput, &muon_i, 0.4));
      displacedTrackIso03Muon1["PATPAT"].push_back(getDisplacedTrackerIsolation<pat::Muon, pat::Muon>(
          generalTracks, muon_i, muonVertex, beamSpotInput, nullptr, 0.3));
      displacedTrackIso04Muon1["PATPAT"].push_back(getDisplacedTrackerIsolation<pat::Muon, pat::Muon>(
          generalTracks, muon_i, muonVertex, beamSpotInput, nullptr, 0.4));
      displacedTrackIso03Muon2["PATPAT"].push_back(getDisplacedTrackerIsolation<pat::Muon, pat::Muon>(
          generalTracks, muon_j, muonVertex, beamSpotInput, nullptr, 0.3));
      displacedTrackIso04Muon2["PATPAT"].push_back(getDisplacedTrackerIsolation<pat::Muon, pat::Muon>(
          generalTracks, muon_j, muonVertex, beamSpotInput, nullptr, 0.4));

      // PAT muons cannot be only tracker muons for proximity deltaR calculations
      if ((muon_i.isGlobalMuon() || muon_i.isStandAloneMuon()) &&
          (muon_j.isGlobalMuon() || muon_j.isStandAloneMuon())) {
        proxDeltaR["PATPAT"].push_back(getProximityDeltaR(*muonTrack_i, *muonTrack_j, magneticField, propagator));
      } else
        proxDeltaR["PATPAT"].push_back(-1);

      DCA["PATPAT"].push_back(std::get<0>(distanceTuple));
      DCAstatus["PATPAT"].push_back(std::get<1>(distanceTuple));
      DCAx["PATPAT"].push_back(std::get<2>(distanceTuple).x());
      DCAy["PATPAT"].push_back(std::get<2>(distanceTuple).y());
      DCAz["PATPAT"].push_back(std::get<2>(distanceTuple).z());

      CheckHitPattern checkHitPattern;
      checkHitPattern.init(tkerTopo, *tkerGeom, builder);
      if (muon_i.isTrackerMuon()) {
        CheckHitPattern::Result hitPattern_i = checkHitPattern(*muonTrack_i, transientMuonVertex.vertexState());
        hitsInFrontOfVert1["PATPAT"].push_back(hitPattern_i.hitsInFrontOfVert);
        missHitsAfterVert1["PATPAT"].push_back(hitPattern_i.missHitsAfterVert);
      } else {
        hitsInFrontOfVert1["PATPAT"].push_back(-1);
        missHitsAfterVert1["PATPAT"].push_back(-1);
      }
      if (muon_j.isTrackerMuon()) {
        CheckHitPattern::Result hitPattern_j = checkHitPattern(*muonTrack_j, transientMuonVertex.vertexState());
        hitsInFrontOfVert2["PATPAT"].push_back(hitPattern_j.hitsInFrontOfVert);
        missHitsAfterVert2["PATPAT"].push_back(hitPattern_j.missHitsAfterVert);
      } else {
        hitsInFrontOfVert2["PATPAT"].push_back(-1);
        missHitsAfterVert2["PATPAT"].push_back(-1);
      }

      reco::TransientTrack refittedTrack_i = transientMuonVertex.refittedTrack(muonTransientTrack_i);
      reco::TransientTrack refittedTrack_j = transientMuonVertex.refittedTrack(muonTransientTrack_j);
      std::vector<reco::TransientTrack> refittedTracks = {refittedTrack_i, refittedTrack_j};
      for (size_t k = 0; k < refittedTracks.size(); ++k) {
        const auto& refittedTrack = refittedTracks[k];
        refittedTrackIsDSAMuon["PATPAT"].push_back(0);
        refittedTrackOriginalIdx["PATPAT"].push_back(i);
        refittedTrackPt["PATPAT"].push_back(refittedTrack.track().pt());
        refittedTrackPtErr["PATPAT"].push_back(refittedTrack.track().ptError());
        refittedTrackPx["PATPAT"].push_back(refittedTrack.track().px());
        refittedTrackPy["PATPAT"].push_back(refittedTrack.track().py());
        refittedTrackPz["PATPAT"].push_back(refittedTrack.track().pz());
        refittedTrackEta["PATPAT"].push_back(refittedTrack.track().eta());
        refittedTrackEtaErr["PATPAT"].push_back(refittedTrack.track().etaError());
        refittedTrackPhi["PATPAT"].push_back(refittedTrack.track().phi());
        refittedTrackPhiErr["PATPAT"].push_back(refittedTrack.track().phiError());
        refittedTrackCharge["PATPAT"].push_back(refittedTrack.track().charge());
        refittedTrackNormChi2["PATPAT"].push_back(refittedTrack.normalizedChi2());
        refittedTrackNdof["PATPAT"].push_back(refittedTrack.ndof());
        refittedTrackChi2["PATPAT"].push_back(refittedTrack.chi2());

        refittedTrackDzPV["PATPAT"].push_back(refittedTrack.track().dz(pv.position()));
        refittedTrackDzPVErr["PATPAT"].push_back(std::hypot(refittedTrack.track().dzError(), pv.zError()));
        TrajectoryStateClosestToPoint trajectoryPV_i = refittedTrack.trajectoryStateClosestToPoint(primaryVertex);
        refittedTrackDxyPVTraj["PATPAT"].push_back(trajectoryPV_i.perigeeParameters().transverseImpactParameter());
        refittedTrackDxyPVTrajErr["PATPAT"].push_back(trajectoryPV_i.perigeeError().transverseImpactParameterError());
        GlobalVector muonRefTrackDir_i(
            refittedTrack.track().px(), refittedTrack.track().py(), refittedTrack.track().pz());
        refittedTrackDxyPVSigned["PATPAT"].push_back(
            IPTools::signedTransverseImpactParameter(refittedTrack, muonRefTrackDir_i, pv).second.value());
        refittedTrackDxyPVSignedErr["PATPAT"].push_back(
            IPTools::signedTransverseImpactParameter(refittedTrack, muonRefTrackDir_i, pv).second.error());
        refittedTrackIp3DPVSigned["PATPAT"].push_back(
            IPTools::signedImpactParameter3D(refittedTrack, muonRefTrackDir_i, pv).second.value());
        refittedTrackIp3DPVSignedErr["PATPAT"].push_back(
            IPTools::signedImpactParameter3D(refittedTrack, muonRefTrackDir_i, pv).second.error());

        refittedTrackIdx["PATPAT"].push_back(ppRefittedTrackIdx_counter);
        if (k == 0)
          refittedTrackIdx1["PATPAT"].push_back(ppRefittedTrackIdx_counter);
        if (k == 1)
          refittedTrackIdx2["PATPAT"].push_back(ppRefittedTrackIdx_counter);
        ppRefittedTrackIdx_counter++;
      }

      refittedTrackIso03Dimuon1["PATPAT"].push_back(getDisplacedTrackerIsolation<reco::Track, reco::Track>(
          generalTracks, refittedTrack_i.track(), muonVertex, beamSpotInput, &refittedTrack_j.track(), 0.3));
      refittedTrackIso04Dimuon1["PATPAT"].push_back(getDisplacedTrackerIsolation<reco::Track, reco::Track>(
          generalTracks, refittedTrack_i.track(), muonVertex, beamSpotInput, &refittedTrack_j.track(), 0.4));
      refittedTrackIso03Dimuon2["PATPAT"].push_back(getDisplacedTrackerIsolation<reco::Track, reco::Track>(
          generalTracks, refittedTrack_j.track(), muonVertex, beamSpotInput, &refittedTrack_i.track(), 0.3));
      refittedTrackIso04Dimuon2["PATPAT"].push_back(getDisplacedTrackerIsolation<reco::Track, reco::Track>(
          generalTracks, refittedTrack_j.track(), muonVertex, beamSpotInput, &refittedTrack_i.track(), 0.4));
      refittedTrackIso03Muon1["PATPAT"].push_back(getDisplacedTrackerIsolation<reco::Track, reco::Track>(
          generalTracks, refittedTrack_i.track(), muonVertex, beamSpotInput, nullptr, 0.3));
      refittedTrackIso04Muon1["PATPAT"].push_back(getDisplacedTrackerIsolation<reco::Track, reco::Track>(
          generalTracks, refittedTrack_i.track(), muonVertex, beamSpotInput, nullptr, 0.4));
      refittedTrackIso03Muon2["PATPAT"].push_back(getDisplacedTrackerIsolation<reco::Track, reco::Track>(
          generalTracks, refittedTrack_j.track(), muonVertex, beamSpotInput, nullptr, 0.3));
      refittedTrackIso04Muon2["PATPAT"].push_back(getDisplacedTrackerIsolation<reco::Track, reco::Track>(
          generalTracks, refittedTrack_j.track(), muonVertex, beamSpotInput, nullptr, 0.4));
    }

    // pat-dsa muon vertex
    for (size_t j = 0; j < dsaMuons.size(); j++) {
      const auto& muonTrack_j = dsaMuons.at(j);
      reco::TransientTrack muonTransientTrack_j = builder.build(muonTrack_j);

      std::vector<reco::TransientTrack> muonTransientTracks{};
      muonTransientTracks.push_back(muonTransientTrack_i);
      muonTransientTracks.push_back(muonTransientTrack_j);

      TransientVertex transientMuonVertex = vertexFitter.vertex(muonTransientTracks);
      reco::Vertex muonVertex = reco::Vertex(transientMuonVertex);

      if (!transientMuonVertex.isValid())
        continue;
      std::tuple<float, float, GlobalPoint> distanceTuple =
          getDistanceBetweenMuonTracks(*muonTrack_i, muonTrack_j, magneticField);
      // if dca status is good but dca is more than 15 cm
      if (std::get<1>(distanceTuple) && std::get<0>(distanceTuple) > 15)
        continue;

      nPatDSAVertices++;

      vertexIsValid["PATDSA"].push_back(transientMuonVertex.isValid());
      std::pair<float, float> vxy_ = getVxy(muonVertex);
      vxy["PATDSA"].push_back(vxy_.first);
      vxySigma["PATDSA"].push_back(vxy_.second);
      std::pair<float, float> vxyz_ = getVxyz(muonVertex);
      vxyz["PATDSA"].push_back(vxyz_.first);
      vxyzSigma["PATDSA"].push_back(vxyz_.second);
      chi2["PATDSA"].push_back(muonVertex.chi2());
      ndof["PATDSA"].push_back(muonVertex.ndof());
      normChi2["PATDSA"].push_back(muonVertex.normalizedChi2());
      vx["PATDSA"].push_back(muonVertex.x());
      vy["PATDSA"].push_back(muonVertex.y());
      vz["PATDSA"].push_back(muonVertex.z());
      t["PATDSA"].push_back(muonVertex.t());
      vxErr["PATDSA"].push_back(muonVertex.xError());
      vyErr["PATDSA"].push_back(muonVertex.yError());
      vzErr["PATDSA"].push_back(muonVertex.zError());
      tErr["PATDSA"].push_back(muonVertex.tError());

      dR["PATDSA"].push_back(reco::deltaR(muon_i.eta(), muonTrack_j.eta(), muon_i.phi(), muonTrack_j.phi()));
      originalMuonIdx1["PATDSA"].push_back(i);
      originalMuonIdx2["PATDSA"].push_back(j);
      isDSAMuon1["PATDSA"].push_back(0);
      isDSAMuon2["PATDSA"].push_back(1);

      displacedTrackIso03Dimuon1["PATDSA"].push_back(getDisplacedTrackerIsolation<pat::Muon, reco::Track>(
          generalTracks, muon_i, muonVertex, beamSpotInput, &muonTrack_j, 0.3));
      displacedTrackIso04Dimuon1["PATDSA"].push_back(getDisplacedTrackerIsolation<pat::Muon, reco::Track>(
          generalTracks, muon_i, muonVertex, beamSpotInput, &muonTrack_j, 0.4));
      displacedTrackIso03Dimuon2["PATDSA"].push_back(getDisplacedTrackerIsolation<reco::Track, pat::Muon>(
          generalTracks, muonTrack_j, muonVertex, beamSpotInput, &muon_i, 0.3));
      displacedTrackIso04Dimuon2["PATDSA"].push_back(getDisplacedTrackerIsolation<reco::Track, pat::Muon>(
          generalTracks, muonTrack_j, muonVertex, beamSpotInput, &muon_i, 0.4));
      displacedTrackIso03Muon1["PATDSA"].push_back(getDisplacedTrackerIsolation<pat::Muon, reco::Track>(
          generalTracks, muon_i, muonVertex, beamSpotInput, nullptr, 0.3));
      displacedTrackIso04Muon1["PATDSA"].push_back(getDisplacedTrackerIsolation<pat::Muon, reco::Track>(
          generalTracks, muon_i, muonVertex, beamSpotInput, nullptr, 0.4));
      displacedTrackIso03Muon2["PATDSA"].push_back(getDisplacedTrackerIsolation<reco::Track, pat::Muon>(
          generalTracks, muonTrack_j, muonVertex, beamSpotInput, nullptr, 0.3));
      displacedTrackIso04Muon2["PATDSA"].push_back(getDisplacedTrackerIsolation<reco::Track, pat::Muon>(
          generalTracks, muonTrack_j, muonVertex, beamSpotInput, nullptr, 0.4));

      // PAT muons cannot be only tracker muons for proximity deltaR calculations
      if (muon_i.isGlobalMuon() || muon_i.isStandAloneMuon()) {
        proxDeltaR["PATDSA"].push_back(getProximityDeltaR(*muonTrack_i, muonTrack_j, magneticField, propagator));
      } else
        proxDeltaR["PATDSA"].push_back(-1);

      DCA["PATDSA"].push_back(std::get<0>(distanceTuple));
      DCAstatus["PATDSA"].push_back(std::get<1>(distanceTuple));
      DCAx["PATDSA"].push_back(std::get<2>(distanceTuple).x());
      DCAy["PATDSA"].push_back(std::get<2>(distanceTuple).y());
      DCAz["PATDSA"].push_back(std::get<2>(distanceTuple).z());

      CheckHitPattern checkHitPattern;
      checkHitPattern.init(tkerTopo, *tkerGeom, builder);
      if (muon_i.isTrackerMuon()) {
        CheckHitPattern::Result hitPattern_i = checkHitPattern(*muonTrack_i, transientMuonVertex.vertexState());
        hitsInFrontOfVert1["PATDSA"].push_back(hitPattern_i.hitsInFrontOfVert);
        missHitsAfterVert1["PATDSA"].push_back(hitPattern_i.missHitsAfterVert);
      } else {
        hitsInFrontOfVert1["PATDSA"].push_back(-1);
        missHitsAfterVert1["PATDSA"].push_back(-1);
      }
      hitsInFrontOfVert2["PATDSA"].push_back(-1);
      missHitsAfterVert2["PATDSA"].push_back(-1);

      reco::TransientTrack refittedTrack_i = transientMuonVertex.refittedTrack(muonTransientTrack_i);
      reco::TransientTrack refittedTrack_j = transientMuonVertex.refittedTrack(muonTransientTrack_j);
      std::vector<reco::TransientTrack> refittedTracks = {refittedTrack_i, refittedTrack_j};
      for (size_t k = 0; k < refittedTracks.size(); ++k) {
        const auto& refittedTrack = refittedTracks[k];
        refittedTrackIsDSAMuon["PATDSA"].push_back(0);
        refittedTrackOriginalIdx["PATDSA"].push_back(i);
        refittedTrackPt["PATDSA"].push_back(refittedTrack.track().pt());
        refittedTrackPtErr["PATDSA"].push_back(refittedTrack.track().ptError());
        refittedTrackPx["PATDSA"].push_back(refittedTrack.track().px());
        refittedTrackPy["PATDSA"].push_back(refittedTrack.track().py());
        refittedTrackPz["PATDSA"].push_back(refittedTrack.track().pz());
        refittedTrackEta["PATDSA"].push_back(refittedTrack.track().eta());
        refittedTrackEtaErr["PATDSA"].push_back(refittedTrack.track().etaError());
        refittedTrackPhi["PATDSA"].push_back(refittedTrack.track().phi());
        refittedTrackPhiErr["PATDSA"].push_back(refittedTrack.track().phiError());
        refittedTrackCharge["PATDSA"].push_back(refittedTrack.track().charge());
        refittedTrackNormChi2["PATDSA"].push_back(refittedTrack.normalizedChi2());
        refittedTrackNdof["PATDSA"].push_back(refittedTrack.ndof());
        refittedTrackChi2["PATDSA"].push_back(refittedTrack.chi2());

        refittedTrackDzPV["PATDSA"].push_back(refittedTrack.track().dz(pv.position()));
        refittedTrackDzPVErr["PATDSA"].push_back(std::hypot(refittedTrack.track().dzError(), pv.zError()));
        TrajectoryStateClosestToPoint trajectoryPV_i = refittedTrack.trajectoryStateClosestToPoint(primaryVertex);
        refittedTrackDxyPVTraj["PATDSA"].push_back(trajectoryPV_i.perigeeParameters().transverseImpactParameter());
        refittedTrackDxyPVTrajErr["PATDSA"].push_back(trajectoryPV_i.perigeeError().transverseImpactParameterError());
        GlobalVector muonRefTrackDir_i(
            refittedTrack.track().px(), refittedTrack.track().py(), refittedTrack.track().pz());
        refittedTrackDxyPVSigned["PATDSA"].push_back(
            IPTools::signedTransverseImpactParameter(refittedTrack, muonRefTrackDir_i, pv).second.value());
        refittedTrackDxyPVSignedErr["PATDSA"].push_back(
            IPTools::signedTransverseImpactParameter(refittedTrack, muonRefTrackDir_i, pv).second.error());
        refittedTrackIp3DPVSigned["PATDSA"].push_back(
            IPTools::signedImpactParameter3D(refittedTrack, muonRefTrackDir_i, pv).second.value());
        refittedTrackIp3DPVSignedErr["PATDSA"].push_back(
            IPTools::signedImpactParameter3D(refittedTrack, muonRefTrackDir_i, pv).second.error());

        refittedTrackIdx["PATDSA"].push_back(pdRefittedTrackIdx_counter);
        if (k == 0)
          refittedTrackIdx1["PATDSA"].push_back(pdRefittedTrackIdx_counter);
        if (k == 1)
          refittedTrackIdx2["PATDSA"].push_back(pdRefittedTrackIdx_counter);
        pdRefittedTrackIdx_counter++;
      }

      refittedTrackIso03Dimuon1["PATDSA"].push_back(getDisplacedTrackerIsolation<reco::Track, reco::Track>(
          generalTracks, refittedTrack_i.track(), muonVertex, beamSpotInput, &refittedTrack_j.track(), 0.3));
      refittedTrackIso04Dimuon1["PATDSA"].push_back(getDisplacedTrackerIsolation<reco::Track, reco::Track>(
          generalTracks, refittedTrack_i.track(), muonVertex, beamSpotInput, &refittedTrack_j.track(), 0.4));
      refittedTrackIso03Dimuon2["PATDSA"].push_back(getDisplacedTrackerIsolation<reco::Track, reco::Track>(
          generalTracks, refittedTrack_j.track(), muonVertex, beamSpotInput, &refittedTrack_i.track(), 0.3));
      refittedTrackIso04Dimuon2["PATDSA"].push_back(getDisplacedTrackerIsolation<reco::Track, reco::Track>(
          generalTracks, refittedTrack_j.track(), muonVertex, beamSpotInput, &refittedTrack_i.track(), 0.4));
      refittedTrackIso03Muon1["PATDSA"].push_back(getDisplacedTrackerIsolation<reco::Track, reco::Track>(
          generalTracks, refittedTrack_i.track(), muonVertex, beamSpotInput, nullptr, 0.3));
      refittedTrackIso04Muon1["PATDSA"].push_back(getDisplacedTrackerIsolation<reco::Track, reco::Track>(
          generalTracks, refittedTrack_i.track(), muonVertex, beamSpotInput, nullptr, 0.4));
      refittedTrackIso03Muon2["PATDSA"].push_back(getDisplacedTrackerIsolation<reco::Track, reco::Track>(
          generalTracks, refittedTrack_j.track(), muonVertex, beamSpotInput, nullptr, 0.3));
      refittedTrackIso04Muon2["PATDSA"].push_back(getDisplacedTrackerIsolation<reco::Track, reco::Track>(
          generalTracks, refittedTrack_j.track(), muonVertex, beamSpotInput, nullptr, 0.4));
    }
  }

  // dsa muons
  for (size_t i = 0; i < dsaMuons.size(); i++) {
    const auto& muonTrack_i = dsaMuons.at(i);
    reco::TransientTrack muonTransientTrack_i = builder.build(muonTrack_i);

    //dsa-dsa muon vertex
    for (size_t j = i + 1; j < dsaMuons.size(); j++) {
      const auto& muonTrack_j = dsaMuons.at(j);
      reco::TransientTrack muonTransientTrack_j = builder.build(muonTrack_j);

      std::vector<reco::TransientTrack> muonTransientTracks{};
      muonTransientTracks.push_back(muonTransientTrack_i);
      muonTransientTracks.push_back(muonTransientTrack_j);

      TransientVertex transientMuonVertex = vertexFitter.vertex(muonTransientTracks);
      reco::Vertex muonVertex = reco::Vertex(transientMuonVertex);

      if (!transientMuonVertex.isValid())
        continue;
      std::tuple<float, float, GlobalPoint> distanceTuple =
          getDistanceBetweenMuonTracks(muonTrack_i, muonTrack_j, magneticField);
      // if dca status is good but dca is more than 15 cm
      if (std::get<1>(distanceTuple) && std::get<0>(distanceTuple) > 15)
        continue;

      nDSADSAVertices++;

      vertexIsValid["DSADSA"].push_back(transientMuonVertex.isValid());

      std::pair<float, float> vxy_ = getVxy(muonVertex);
      vxy["DSADSA"].push_back(vxy_.first);
      vxySigma["DSADSA"].push_back(vxy_.second);
      std::pair<float, float> vxyz_ = getVxyz(muonVertex);
      vxyz["DSADSA"].push_back(vxyz_.first);
      vxyzSigma["DSADSA"].push_back(vxyz_.second);
      chi2["DSADSA"].push_back(muonVertex.chi2());
      ndof["DSADSA"].push_back(muonVertex.ndof());
      normChi2["DSADSA"].push_back(muonVertex.normalizedChi2());
      vx["DSADSA"].push_back(muonVertex.x());
      vy["DSADSA"].push_back(muonVertex.y());
      vz["DSADSA"].push_back(muonVertex.z());
      t["DSADSA"].push_back(muonVertex.t());
      vxErr["DSADSA"].push_back(muonVertex.xError());
      vyErr["DSADSA"].push_back(muonVertex.yError());
      vzErr["DSADSA"].push_back(muonVertex.zError());
      tErr["DSADSA"].push_back(muonVertex.tError());
      dR["DSADSA"].push_back(reco::deltaR(muonTrack_i.eta(), muonTrack_j.eta(), muonTrack_i.phi(), muonTrack_j.phi()));
      originalMuonIdx1["DSADSA"].push_back(i);
      originalMuonIdx2["DSADSA"].push_back(j);
      isDSAMuon1["DSADSA"].push_back(1);
      isDSAMuon2["DSADSA"].push_back(1);

      displacedTrackIso03Dimuon1["DSADSA"].push_back(getDisplacedTrackerIsolation<reco::Track, reco::Track>(
          generalTracks, muonTrack_i, muonVertex, beamSpotInput, &muonTrack_j, 0.3));
      displacedTrackIso04Dimuon1["DSADSA"].push_back(getDisplacedTrackerIsolation<reco::Track, reco::Track>(
          generalTracks, muonTrack_i, muonVertex, beamSpotInput, &muonTrack_j, 0.4));
      displacedTrackIso03Dimuon2["DSADSA"].push_back(getDisplacedTrackerIsolation<reco::Track, reco::Track>(
          generalTracks, muonTrack_j, muonVertex, beamSpotInput, &muonTrack_i, 0.3));
      displacedTrackIso04Dimuon2["DSADSA"].push_back(getDisplacedTrackerIsolation<reco::Track, reco::Track>(
          generalTracks, muonTrack_j, muonVertex, beamSpotInput, &muonTrack_i, 0.4));
      displacedTrackIso03Muon1["DSADSA"].push_back(getDisplacedTrackerIsolation<reco::Track, reco::Track>(
          generalTracks, muonTrack_i, muonVertex, beamSpotInput, nullptr, 0.3));
      displacedTrackIso04Muon1["DSADSA"].push_back(getDisplacedTrackerIsolation<reco::Track, reco::Track>(
          generalTracks, muonTrack_i, muonVertex, beamSpotInput, nullptr, 0.4));
      displacedTrackIso03Muon2["DSADSA"].push_back(getDisplacedTrackerIsolation<reco::Track, reco::Track>(
          generalTracks, muonTrack_j, muonVertex, beamSpotInput, nullptr, 0.3));
      displacedTrackIso04Muon2["DSADSA"].push_back(getDisplacedTrackerIsolation<reco::Track, reco::Track>(
          generalTracks, muonTrack_j, muonVertex, beamSpotInput, nullptr, 0.4));

      proxDeltaR["DSADSA"].push_back(getProximityDeltaR(muonTrack_i, muonTrack_j, magneticField, propagator));

      DCA["DSADSA"].push_back(std::get<0>(distanceTuple));
      DCAstatus["DSADSA"].push_back(std::get<1>(distanceTuple));
      DCAx["DSADSA"].push_back(std::get<2>(distanceTuple).x());
      DCAy["DSADSA"].push_back(std::get<2>(distanceTuple).y());
      DCAz["DSADSA"].push_back(std::get<2>(distanceTuple).z());

      hitsInFrontOfVert1["DSADSA"].push_back(-1);
      missHitsAfterVert1["DSADSA"].push_back(-1);
      hitsInFrontOfVert2["DSADSA"].push_back(-1);
      missHitsAfterVert2["DSADSA"].push_back(-1);

      reco::TransientTrack refittedTrack_i = transientMuonVertex.refittedTrack(muonTransientTrack_i);
      reco::TransientTrack refittedTrack_j = transientMuonVertex.refittedTrack(muonTransientTrack_j);
      std::vector<reco::TransientTrack> refittedTracks = {refittedTrack_i, refittedTrack_j};
      for (size_t k = 0; k < refittedTracks.size(); ++k) {
        const auto& refittedTrack = refittedTracks[k];
        refittedTrackIsDSAMuon["DSADSA"].push_back(0);
        refittedTrackOriginalIdx["DSADSA"].push_back(i);
        refittedTrackPt["DSADSA"].push_back(refittedTrack.track().pt());
        refittedTrackPtErr["DSADSA"].push_back(refittedTrack.track().ptError());
        refittedTrackPx["DSADSA"].push_back(refittedTrack.track().px());
        refittedTrackPy["DSADSA"].push_back(refittedTrack.track().py());
        refittedTrackPz["DSADSA"].push_back(refittedTrack.track().pz());
        refittedTrackEta["DSADSA"].push_back(refittedTrack.track().eta());
        refittedTrackEtaErr["DSADSA"].push_back(refittedTrack.track().etaError());
        refittedTrackPhi["DSADSA"].push_back(refittedTrack.track().phi());
        refittedTrackPhiErr["DSADSA"].push_back(refittedTrack.track().phiError());
        refittedTrackCharge["DSADSA"].push_back(refittedTrack.track().charge());
        refittedTrackNormChi2["DSADSA"].push_back(refittedTrack.normalizedChi2());
        refittedTrackNdof["DSADSA"].push_back(refittedTrack.ndof());
        refittedTrackChi2["DSADSA"].push_back(refittedTrack.chi2());

        refittedTrackDzPV["DSADSA"].push_back(refittedTrack.track().dz(pv.position()));
        refittedTrackDzPVErr["DSADSA"].push_back(std::hypot(refittedTrack.track().dzError(), pv.zError()));
        TrajectoryStateClosestToPoint trajectoryPV_i = refittedTrack.trajectoryStateClosestToPoint(primaryVertex);
        refittedTrackDxyPVTraj["DSADSA"].push_back(trajectoryPV_i.perigeeParameters().transverseImpactParameter());
        refittedTrackDxyPVTrajErr["DSADSA"].push_back(trajectoryPV_i.perigeeError().transverseImpactParameterError());
        GlobalVector muonRefTrackDir_i(
            refittedTrack.track().px(), refittedTrack.track().py(), refittedTrack.track().pz());
        refittedTrackDxyPVSigned["DSADSA"].push_back(
            IPTools::signedTransverseImpactParameter(refittedTrack, muonRefTrackDir_i, pv).second.value());
        refittedTrackDxyPVSignedErr["DSADSA"].push_back(
            IPTools::signedTransverseImpactParameter(refittedTrack, muonRefTrackDir_i, pv).second.error());
        refittedTrackIp3DPVSigned["DSADSA"].push_back(
            IPTools::signedImpactParameter3D(refittedTrack, muonRefTrackDir_i, pv).second.value());
        refittedTrackIp3DPVSignedErr["DSADSA"].push_back(
            IPTools::signedImpactParameter3D(refittedTrack, muonRefTrackDir_i, pv).second.error());

        refittedTrackIdx["DSADSA"].push_back(ddRefittedTrackIdx_counter);
        if (k == 0)
          refittedTrackIdx1["DSADSA"].push_back(ddRefittedTrackIdx_counter);
        if (k == 1)
          refittedTrackIdx2["DSADSA"].push_back(ddRefittedTrackIdx_counter);
        ddRefittedTrackIdx_counter++;
      }

      refittedTrackIso03Dimuon1["DSADSA"].push_back(getDisplacedTrackerIsolation<reco::Track, reco::Track>(
          generalTracks, refittedTrack_i.track(), muonVertex, beamSpotInput, &refittedTrack_j.track(), 0.3));
      refittedTrackIso04Dimuon1["DSADSA"].push_back(getDisplacedTrackerIsolation<reco::Track, reco::Track>(
          generalTracks, refittedTrack_i.track(), muonVertex, beamSpotInput, &refittedTrack_j.track(), 0.4));
      refittedTrackIso03Dimuon2["DSADSA"].push_back(getDisplacedTrackerIsolation<reco::Track, reco::Track>(
          generalTracks, refittedTrack_j.track(), muonVertex, beamSpotInput, &refittedTrack_i.track(), 0.3));
      refittedTrackIso04Dimuon2["DSADSA"].push_back(getDisplacedTrackerIsolation<reco::Track, reco::Track>(
          generalTracks, refittedTrack_j.track(), muonVertex, beamSpotInput, &refittedTrack_i.track(), 0.4));
      refittedTrackIso03Muon1["DSADSA"].push_back(getDisplacedTrackerIsolation<reco::Track, reco::Track>(
          generalTracks, refittedTrack_i.track(), muonVertex, beamSpotInput, nullptr, 0.3));
      refittedTrackIso04Muon1["DSADSA"].push_back(getDisplacedTrackerIsolation<reco::Track, reco::Track>(
          generalTracks, refittedTrack_i.track(), muonVertex, beamSpotInput, nullptr, 0.4));
      refittedTrackIso03Muon2["DSADSA"].push_back(getDisplacedTrackerIsolation<reco::Track, reco::Track>(
          generalTracks, refittedTrack_j.track(), muonVertex, beamSpotInput, nullptr, 0.3));
      refittedTrackIso04Muon2["DSADSA"].push_back(getDisplacedTrackerIsolation<reco::Track, reco::Track>(
          generalTracks, refittedTrack_j.track(), muonVertex, beamSpotInput, nullptr, 0.4));
    }
  }

  auto patVertexTab = std::make_unique<nanoaod::FlatTable>(nPatPatVertices, "PatMuonVertex", false, false);
  auto patdsaVertexTab = std::make_unique<nanoaod::FlatTable>(nPatDSAVertices, "PatDSAMuonVertex", false, false);
  auto dsaVertexTab = std::make_unique<nanoaod::FlatTable>(nDSADSAVertices, "DSAMuonVertex", false, false);

  std::map<std::string, std::unique_ptr<nanoaod::FlatTable>> vertexTables;
  vertexTables["PATPAT"] = std::move(patVertexTab);
  vertexTables["PATDSA"] = std::move(patdsaVertexTab);
  vertexTables["DSADSA"] = std::move(dsaVertexTab);

  for (const auto& [key, table] : vertexTables) {
    table->addColumn<float>("isValid", vertexIsValid[key], "");
    table->addColumn<float>("vxy", vxy[key], "");
    table->addColumn<float>("vxySigma", vxySigma[key], "");
    table->addColumn<float>("vxyz", vxyz[key], "");
    table->addColumn<float>("vxyzSigma", vxyzSigma[key], "");
    table->addColumn<float>("chi2", chi2[key], "");
    table->addColumn<float>("ndof", ndof[key], "");
    table->addColumn<float>("normChi2", normChi2[key], "");
    table->addColumn<float>("vx", vx[key], "");
    table->addColumn<float>("vy", vy[key], "");
    table->addColumn<float>("vz", vz[key], "");
    table->addColumn<float>("t", t[key], "");
    table->addColumn<float>("vxErr", vxErr[key], "");
    table->addColumn<float>("vyErr", vyErr[key], "");
    table->addColumn<float>("vzErr", vzErr[key], "");
    table->addColumn<float>("tErr", tErr[key], "");
    table->addColumn<float>("dR", dR[key], "");
    table->addColumn<float>("originalMuonIdx1", originalMuonIdx1[key], "");
    table->addColumn<float>("originalMuonIdx2", originalMuonIdx2[key], "");
    table->addColumn<float>("isDSAMuon1", isDSAMuon1[key], "");
    table->addColumn<float>("isDSAMuon2", isDSAMuon2[key], "");
    table->addColumn<float>("displacedTrackIso03Dimuon1", displacedTrackIso03Dimuon1[key], "");
    table->addColumn<float>("displacedTrackIso04Dimuon1", displacedTrackIso04Dimuon1[key], "");
    table->addColumn<float>("displacedTrackIso03Dimuon2", displacedTrackIso03Dimuon2[key], "");
    table->addColumn<float>("displacedTrackIso04Dimuon2", displacedTrackIso04Dimuon2[key], "");
    table->addColumn<float>("displacedTrackIso03Muon1", displacedTrackIso03Muon1[key], "");
    table->addColumn<float>("displacedTrackIso04Muon1", displacedTrackIso04Muon1[key], "");
    table->addColumn<float>("displacedTrackIso03Muon2", displacedTrackIso03Muon2[key], "");
    table->addColumn<float>("displacedTrackIso04Muon2", displacedTrackIso04Muon2[key], "");
    table->addColumn<float>("dRprox", proxDeltaR[key], "");
    table->addColumn<float>("dca", DCA[key], "");
    table->addColumn<float>("dcaStatus", DCAstatus[key], "");
    table->addColumn<float>("dcax", DCAx[key], "");
    table->addColumn<float>("dcay", DCAy[key], "");
    table->addColumn<float>("dcaz", DCAz[key], "");
    table->addColumn<float>("hitsInFrontOfVert1", hitsInFrontOfVert1[key], "");
    table->addColumn<float>("hitsInFrontOfVert2", hitsInFrontOfVert2[key], "");
    table->addColumn<float>("missHitsAfterVert1", missHitsAfterVert1[key], "");
    table->addColumn<float>("missHitsAfterVert2", missHitsAfterVert2[key], "");
    table->addColumn<float>("refittedTrackIdx1", refittedTrackIdx1[key], "");
    table->addColumn<float>("refittedTrackIdx2", refittedTrackIdx2[key], "");
    table->addColumn<float>("refittedTrackIso03Dimuon1", refittedTrackIso03Dimuon1[key], "");
    table->addColumn<float>("refittedTrackIso04Dimuon1", refittedTrackIso04Dimuon1[key], "");
    table->addColumn<float>("refittedTrackIso03Dimuon2", refittedTrackIso03Dimuon2[key], "");
    table->addColumn<float>("refittedTrackIso04Dimuon2", refittedTrackIso04Dimuon2[key], "");
    table->addColumn<float>("refittedTrackIso03Muon1", refittedTrackIso03Muon1[key], "");
    table->addColumn<float>("refittedTrackIso04Muon1", refittedTrackIso04Muon1[key], "");
    table->addColumn<float>("refittedTrackIso03Muon2", refittedTrackIso03Muon2[key], "");
    table->addColumn<float>("refittedTrackIso04Muon2", refittedTrackIso04Muon2[key], "");
  }

  iEvent.put(std::move(vertexTables["PATPAT"]), "PatMuonVertex");
  iEvent.put(std::move(vertexTables["PATDSA"]), "PatDSAMuonVertex");
  iEvent.put(std::move(vertexTables["DSADSA"]), "DSAMuonVertex");

  auto patRefittedTracksTab =
      std::make_unique<nanoaod::FlatTable>(nPatPatVertices * 2, "PatMuonVertexRefittedTracks", false, false);
  auto patdsaRefittedTracksTab =
      std::make_unique<nanoaod::FlatTable>(nPatDSAVertices * 2, "PatDSAMuonVertexRefittedTracks", false, false);
  auto dsaRefittedTracksTab =
      std::make_unique<nanoaod::FlatTable>(nDSADSAVertices * 2, "DSAMuonVertexRefittedTracks", false, false);

  std::map<std::string, std::unique_ptr<nanoaod::FlatTable>> refittedTracksTables;
  refittedTracksTables["PATPAT"] = std::move(patRefittedTracksTab);
  refittedTracksTables["PATDSA"] = std::move(patdsaRefittedTracksTab);
  refittedTracksTables["DSADSA"] = std::move(dsaRefittedTracksTab);

  for (const auto& [key, table] : refittedTracksTables) {
    table->addColumn<float>("idx", refittedTrackIdx[key], "");
    table->addColumn<float>("isDSAMuon", refittedTrackIsDSAMuon[key], "");
    table->addColumn<float>("originalMuonIdx", refittedTrackOriginalIdx[key], "");
    table->addColumn<float>("pt", refittedTrackPt[key], "");
    table->addColumn<float>("ptErr", refittedTrackPtErr[key], "");
    table->addColumn<float>("px", refittedTrackPx[key], "");
    table->addColumn<float>("py", refittedTrackPy[key], "");
    table->addColumn<float>("pz", refittedTrackPz[key], "");
    table->addColumn<float>("eta", refittedTrackEta[key], "");
    table->addColumn<float>("etaErr", refittedTrackEtaErr[key], "");
    table->addColumn<float>("phi", refittedTrackPhi[key], "");
    table->addColumn<float>("phiErr", refittedTrackPhiErr[key], "");
    table->addColumn<float>("charge", refittedTrackCharge[key], "");
    table->addColumn<float>("normChi2", refittedTrackNormChi2[key], "");
    table->addColumn<float>("ndof", refittedTrackNdof[key], "");
    table->addColumn<float>("chi2", refittedTrackChi2[key], "");
    table->addColumn<float>("dzPV", refittedTrackDzPV[key], "");
    table->addColumn<float>("dzPVErr", refittedTrackDzPVErr[key], "");
    table->addColumn<float>("dxyPVTraj", refittedTrackDxyPVTraj[key], "");
    table->addColumn<float>("dxyPVTrajErr", refittedTrackDxyPVTrajErr[key], "");
    table->addColumn<float>("dxyPVSigned", refittedTrackDxyPVSigned[key], "");
    table->addColumn<float>("dxyPVSignedErr", refittedTrackDxyPVSignedErr[key], "");
    table->addColumn<float>("ip3DPVSigned", refittedTrackIp3DPVSigned[key], "");
    table->addColumn<float>("ip3DPVSignedErr", refittedTrackIp3DPVSignedErr[key], "");
  }

  iEvent.put(std::move(refittedTracksTables["PATPAT"]), "PatMuonVertexRefittedTracks");
  iEvent.put(std::move(refittedTracksTables["PATDSA"]), "PatDSAMuonVertexRefittedTracks");
  iEvent.put(std::move(refittedTracksTables["DSADSA"]), "DSAMuonVertexRefittedTracks");
}

std::pair<float, float> MuonVertexTableProducer::getVxy(const reco::Vertex muonVertex) const {
  float vxy = sqrt(muonVertex.x() * muonVertex.x() + muonVertex.y() * muonVertex.y());
  float vxySigma = (1 / vxy) * sqrt(muonVertex.x() * muonVertex.x() * muonVertex.xError() * muonVertex.xError() +
                                    muonVertex.y() * muonVertex.y() * muonVertex.yError() * muonVertex.yError());
  return std::make_pair(vxy, vxySigma);
}

std::pair<float, float> MuonVertexTableProducer::getVxyz(const reco::Vertex muonVertex) const {
  float vxyz =
      sqrt(muonVertex.x() * muonVertex.x() + muonVertex.y() * muonVertex.y() + muonVertex.z() * muonVertex.z());
  float vxyzSigma = (1 / vxyz) * sqrt(muonVertex.x() * muonVertex.x() * muonVertex.xError() * muonVertex.xError() +
                                      muonVertex.y() * muonVertex.y() * muonVertex.yError() * muonVertex.yError() +
                                      muonVertex.z() * muonVertex.z() * muonVertex.zError() * muonVertex.zError());
  return std::make_pair(vxyz, vxyzSigma);
}

template <typename MuonType1, typename MuonType2>
float MuonVertexTableProducer::getDisplacedTrackerIsolation(const std::vector<reco::Track>& generalTracks,
                                                            const MuonType1& muon_1,
                                                            const reco::Vertex muonVertex,
                                                            const reco::BeamSpot& beamspot,
                                                            const MuonType2* muon_2,
                                                            float maxDR,
                                                            float minDR,
                                                            float maxDz,
                                                            float maxDxy) const {
  float trackPtSum = 0;

  int nGeneralTracks = generalTracks.size();
  float muonTrack2_pt = 0;
  float muonTrack2_minDR = 9999;

  for (int i = 0; i < nGeneralTracks; i++) {
    const reco::Track& generalTrack = (generalTracks)[i];

    // Muon POG Tracker Isolation recommendation
    float dR = deltaR(muon_1.eta(), muon_1.phi(), generalTrack.eta(), generalTrack.phi());
    if (dR > maxDR)
      continue;
    if (abs(generalTrack.vz() - muonVertex.z()) > maxDz)
      continue;
    if (generalTrack.dxy(beamspot) > maxDxy)
      continue;
    if (dR < minDR)
      continue;

    // Determine if track belongs to other muon and get pt of the track
    // Only if muon is given as input
    if (muon_2 != nullptr) {
      float dR_2 = deltaR(muon_2->eta(), muon_2->phi(), generalTrack.eta(), generalTrack.phi());
      if (dR_2 < minDR && dR_2 < muonTrack2_minDR) {
        muonTrack2_pt = generalTrack.pt();
        muonTrack2_minDR = dR_2;
      }
    }

    trackPtSum += generalTrack.pt();
  }

  // Remove pt of track that belongs to other muon
  trackPtSum -= muonTrack2_pt;

  float ptRatio = trackPtSum / muon_1.pt();
  return ptRatio;
}

/**
*  Proximity match based on EXO-23-010
*  Calculating deltaR between the innermost hit of the DSAMuon trackRef
*  and the extracted closest position of PATMuon track
**/
template <typename MuonType>
float MuonVertexTableProducer::getProximityDeltaR(const MuonType& track,
                                                  const MuonType& trackRef,
                                                  const edm::ESHandle<MagneticField>& magneticField,
                                                  const edm::ESHandle<Propagator>& propagator) const {
  FreeTrajectoryState trajState(GlobalPoint(track.vx(), track.vy(), track.vz()),
                                GlobalVector(track.px(), track.py(), track.pz()),
                                track.charge(),
                                magneticField.product());

  GlobalPoint refPos(trackRef.innerPosition().x(), trackRef.innerPosition().y(), trackRef.innerPosition().z());
  FreeTrajectoryState trajStatePCA(propagator->propagate(trajState, refPos));

  float dR = deltaR(trajStatePCA.position().eta(),
                    trajStatePCA.position().phi(),
                    trackRef.innerPosition().eta(),
                    trackRef.innerPosition().phi());
  return dR;
}

/**
*  Proximity between the muons based on EXO-23-010
*  Getting Distance of Closest Approach between muon tracks using TwoTrackMinimumDistance
*  Returns tuple of distance (float), error of distance (float) and crossing point (GlobalPoint)
**/
std::tuple<float, float, GlobalPoint> MuonVertexTableProducer::getDistanceBetweenMuonTracks(
    const reco::Track& track1, const reco::Track& track2, const edm::ESHandle<MagneticField>& magneticField) const {
  TwoTrackMinimumDistance ttmd;
  FreeTrajectoryState fts1(GlobalPoint(track1.vx(), track1.vy(), track1.vz()),
                           GlobalVector(track1.px(), track1.py(), track1.pz()),
                           track1.charge(),
                           magneticField.product());
  FreeTrajectoryState fts2(GlobalPoint(track2.vx(), track2.vy(), track2.vz()),
                           GlobalVector(track2.px(), track2.py(), track2.pz()),
                           track2.charge(),
                           magneticField.product());
  bool status = ttmd.calculate(fts1, fts2);
  if (!status)
    return std::tuple(-999.f, status, GlobalPoint(-999.f, -999.f, -999.f));
  return std::make_tuple(ttmd.distance(), status, ttmd.crossingPoint());
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MuonVertexTableProducer);
