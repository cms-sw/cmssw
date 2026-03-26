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
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "TLorentzVector.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/PatCandidates/interface/Electron.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"

#include "RecoVertex/KinematicFitPrimitives/interface/ParticleMass.h"
#include "RecoVertex/KinematicFitPrimitives/interface/MultiTrackKinematicConstraint.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicParticleFactoryFromTransientTrack.h"
#include "RecoVertex/KinematicFit/interface/KinematicConstrainedVertexFitter.h"
#include "RecoVertex/KinematicFit/interface/TwoTrackMassKinematicConstraint.h"
#include "RecoVertex/KinematicFit/interface/KinematicParticleVertexFitter.h"
#include "RecoVertex/KinematicFit/interface/KinematicParticleFitter.h"
#include "RecoVertex/KinematicFit/interface/MassKinematicConstraint.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

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

class ElectronVertexTableProducer : public edm::global::EDProducer<> {
public:
  explicit ElectronVertexTableProducer(const edm::ParameterSet& iConfig)
      : electronTag_(consumes<std::vector<pat::Electron>>(iConfig.getParameter<edm::InputTag>("electrons"))),
        bsTag_(consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamspot"))),
        pvTag_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("primaryVertex"))),
        magneticFieldToken_(esConsumes<MagneticField, IdealMagneticFieldRecord>()),
        tkerGeomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>()),
        tkerTopoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd>()),
        transientTrackBuilderToken_(esConsumes(edm::ESInputTag("", "TransientTrackBuilder"))) {
    produces<nanoaod::FlatTable>("ElectronVertex");
    produces<nanoaod::FlatTable>("ElectronVertexRefittedTracks");
  }

  ~ElectronVertexTableProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("electrons")->setComment("input pat electrons collection");
    desc.add<edm::InputTag>("beamspot")->setComment("input beamspot collection");
    desc.add<edm::InputTag>("primaryVertex")->setComment("input primaryVertex collection");
    descriptions.add("electronVertexTables", desc);
  }

private:
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

  std::pair<float, float> getVxy(const reco::Vertex electronVertex) const;
  std::pair<float, float> getVxyz(const reco::Vertex electronVertex) const;

  std::tuple<float, float, GlobalPoint> getDistanceBetweenElectronTracks(
      const reco::Track& track1, const reco::Track& track2, const edm::ESHandle<MagneticField>& magneticField) const;

  const edm::EDGetTokenT<std::vector<pat::Electron>> electronTag_;
  const edm::EDGetTokenT<reco::BeamSpot> bsTag_;
  const edm::EDGetTokenT<reco::VertexCollection> pvTag_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magneticFieldToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> tkerGeomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tkerTopoToken_;
  const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> transientTrackBuilderToken_;
};

void ElectronVertexTableProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  const std::vector<pat::Electron>& electrons = iEvent.get(electronTag_);

  const reco::BeamSpot& beamSpotInput = iEvent.get(bsTag_);
  const auto& bs = beamSpotInput.position();
  GlobalPoint beamSpot(bs.x(), bs.y(), bs.z());
  reco::Vertex beamSpotVertex(beamSpotInput.position(), beamSpotInput.covariance3D());

  const reco::VertexCollection& primaryVertices = iEvent.get(pvTag_);
  const auto& pv = primaryVertices.at(0);
  GlobalPoint primaryVertex(pv.x(), pv.y(), pv.z());

  auto const& magneticField = &iSetup.getData(magneticFieldToken_);

  auto const& tkerGeom = &iSetup.getData(tkerGeomToken_);
  auto const& tkerTopo = &iSetup.getData(tkerTopoToken_);
  const TransientTrackBuilder& builder = iSetup.getData(transientTrackBuilderToken_);

  KalmanVertexFitter vertexFitter(true, true);

  int nElectronVertices = 0;
  int refittedTrackIdx_counter = 0;

  std::vector<bool> vertexIsValid;
  std::vector<float> vxy, vxyz, vx, vy, vz, t, vxySigma, vxyzSigma, vxErr, vyErr, vzErr, tErr;
  std::vector<float> chi2, ndof, normChi2, dR, originalElectronIdx1, originalElectronIdx2, refittedTrackIdx1,
      refittedTrackIdx2;
  std::vector<float> DCA, DCAstatus, DCAx, DCAy, DCAz;
  std::vector<float> hitsInFrontOfVert1, missHitsAfterVert1, hitsInFrontOfVert2, missHitsAfterVert2;

  std::vector<float> refittedTrackIdx, refittedTrackOriginalIdx, refittedTrackPt, refittedTrackPtErr, refittedTrackPx,
      refittedTrackPy, refittedTrackPz;
  std::vector<float> refittedTrackEta, refittedTrackEtaErr, refittedTrackPhi, refittedTrackPhiErr, refittedTrackCharge,
      refittedTrackNormChi2, refittedTrackNdof, refittedTrackChi2;
  std::vector<float> refittedTrackDzPV, refittedTrackDzPVErr, refittedTrackDxyPVTraj, refittedTrackDxyPVTrajErr,
      refittedTrackDxyPVSigned, refittedTrackDxyPVSignedErr, refittedTrackIp3DPVSigned, refittedTrackIp3DPVSignedErr;
  std::vector<float> refittedTrackDxyBS, refittedTrackDxyBSErr, refittedTrackDzBS, refittedTrackDzBSErr,
      refittedTrackDxyBSTraj, refittedTrackDxyBSTrajErr, refittedTrackDxyBSSigned, refittedTrackDxyBSSignedErr,
      refittedTrackIp3DBSSigned, refittedTrackIp3DBSSignedErr;
  std::vector<float> refittedVertMass, refittedVertPt, refittedVertEta, refittedVertPhi;
  // pat electrons
  for (size_t i = 0; i < electrons.size(); i++) {
    const pat::Electron& electron_i = electrons.at(i);

    reco::GsfTrackRef trackRef_i = electron_i.gsfTrack();

    const auto& electronTrack_i = trackRef_i.get();
    reco::TransientTrack electronTransientTrack_i = builder.build(electronTrack_i);

    // pat-pat electron vertex
    for (size_t j = i + 1; j < electrons.size(); j++) {
      const pat::Electron& electron_j = electrons.at(j);

      reco::GsfTrackRef trackRef_j = electron_j.gsfTrack();

      const auto& electronTrack_j = trackRef_j.get();
      reco::TransientTrack electronTransientTrack_j = builder.build(electronTrack_j);

      std::vector<reco::TransientTrack> electronTransientTracks{};
      electronTransientTracks.push_back(electronTransientTrack_i);
      electronTransientTracks.push_back(electronTransientTrack_j);

      TransientVertex transientElectronVertex = vertexFitter.vertex(electronTransientTracks);
      reco::Vertex electronVertex = reco::Vertex(transientElectronVertex);

      if (!transientElectronVertex.isValid())
        continue;
      std::tuple<float, float, GlobalPoint> distanceTuple =
          getDistanceBetweenElectronTracks(*electronTrack_i, *electronTrack_j, magneticField);
      // if dca status is good but dca is more than 15 cm
      if (std::get<1>(distanceTuple) && std::get<0>(distanceTuple) > 15)
        continue;

      nElectronVertices++;

      vertexIsValid.push_back(transientElectronVertex.isValid());
      std::pair<float, float> vxy_ = getVxy(electronVertex);
      vxy.push_back(vxy_.first);
      vxySigma.push_back(vxy_.second);
      std::pair<float, float> vxyz_ = getVxyz(electronVertex);
      vxyz.push_back(vxyz_.first);
      vxyzSigma.push_back(vxyz_.second);
      ndof.push_back(electronVertex.ndof());
      normChi2.push_back(electronVertex.normalizedChi2());
      chi2.push_back(electronVertex.chi2());
      vx.push_back(electronVertex.x());
      vy.push_back(electronVertex.y());
      vz.push_back(electronVertex.z());
      t.push_back(electronVertex.t());
      vxErr.push_back(electronVertex.xError());
      vyErr.push_back(electronVertex.yError());
      vzErr.push_back(electronVertex.zError());
      tErr.push_back(electronVertex.tError());

      dR.push_back(reco::deltaR(electron_i, electron_j));
      originalElectronIdx1.push_back(i);
      originalElectronIdx2.push_back(j);

      DCA.push_back(std::get<0>(distanceTuple));
      DCAstatus.push_back(std::get<1>(distanceTuple));
      DCAx.push_back(std::get<2>(distanceTuple).x());
      DCAy.push_back(std::get<2>(distanceTuple).y());
      DCAz.push_back(std::get<2>(distanceTuple).z());

      CheckHitPattern checkHitPattern;
      checkHitPattern.init(tkerTopo, *tkerGeom, builder);
      CheckHitPattern::Result hitPattern_i = checkHitPattern(*electronTrack_i, transientElectronVertex.vertexState());
      hitsInFrontOfVert1.push_back(hitPattern_i.hitsInFrontOfVert);
      missHitsAfterVert1.push_back(hitPattern_i.missHitsAfterVert);

      CheckHitPattern::Result hitPattern_j = checkHitPattern(*electronTrack_j, transientElectronVertex.vertexState());
      hitsInFrontOfVert2.push_back(hitPattern_j.hitsInFrontOfVert);
      missHitsAfterVert2.push_back(hitPattern_j.missHitsAfterVert);

      reco::TransientTrack refittedTrack_i = transientElectronVertex.refittedTrack(electronTransientTrack_i);
      reco::TransientTrack refittedTrack_j = transientElectronVertex.refittedTrack(electronTransientTrack_j);
      refittedTrackOriginalIdx.push_back(i);
      refittedTrackPt.push_back(refittedTrack_i.track().pt());
      refittedTrackPtErr.push_back(refittedTrack_i.track().ptError());
      refittedTrackPx.push_back(refittedTrack_i.track().px());
      refittedTrackPy.push_back(refittedTrack_i.track().py());
      refittedTrackPz.push_back(refittedTrack_i.track().pz());
      refittedTrackEta.push_back(refittedTrack_i.track().eta());
      refittedTrackEtaErr.push_back(refittedTrack_i.track().etaError());
      refittedTrackPhi.push_back(refittedTrack_i.track().phi());
      refittedTrackPhiErr.push_back(refittedTrack_i.track().phiError());
      refittedTrackCharge.push_back(refittedTrack_i.track().charge());
      refittedTrackNormChi2.push_back(refittedTrack_i.normalizedChi2());
      refittedTrackNdof.push_back(refittedTrack_i.ndof());
      refittedTrackChi2.push_back(refittedTrack_i.chi2());

      refittedTrackDzPV.push_back(refittedTrack_i.track().dz(pv.position()));
      refittedTrackDzPVErr.push_back(std::hypot(refittedTrack_i.track().dzError(), pv.zError()));
      TrajectoryStateClosestToPoint trajectoryPV_i = refittedTrack_i.trajectoryStateClosestToPoint(primaryVertex);
      refittedTrackDxyPVTraj.push_back(trajectoryPV_i.perigeeParameters().transverseImpactParameter());
      refittedTrackDxyPVTrajErr.push_back(trajectoryPV_i.perigeeError().transverseImpactParameterError());
      GlobalVector electronRefTrackDir_i(
          refittedTrack_i.track().px(), refittedTrack_i.track().py(), refittedTrack_i.track().pz());
      refittedTrackDxyPVSigned.push_back(
          IPTools::signedTransverseImpactParameter(refittedTrack_i, electronRefTrackDir_i, pv).second.value());
      refittedTrackDxyPVSignedErr.push_back(
          IPTools::signedTransverseImpactParameter(refittedTrack_i, electronRefTrackDir_i, pv).second.error());
      refittedTrackIp3DPVSigned.push_back(
          IPTools::signedImpactParameter3D(refittedTrack_i, electronRefTrackDir_i, pv).second.value());
      refittedTrackIp3DPVSignedErr.push_back(
          IPTools::signedImpactParameter3D(refittedTrack_i, electronRefTrackDir_i, pv).second.error());
      refittedTrackDxyBS.push_back(refittedTrack_i.track().dxy(bs));
      refittedTrackDxyBSErr.push_back(std::hypot(refittedTrack_i.track().dxyError(), beamSpotVertex.zError()));
      refittedTrackDzBS.push_back(refittedTrack_i.track().dz(bs));
      refittedTrackDzBSErr.push_back(std::hypot(refittedTrack_i.track().dzError(), beamSpotVertex.zError()));
      TrajectoryStateClosestToBeamLine trajectoryBS_i = refittedTrack_i.stateAtBeamLine();
      refittedTrackDxyBSTraj.push_back(trajectoryBS_i.transverseImpactParameter().value());
      refittedTrackDxyBSTrajErr.push_back(trajectoryBS_i.transverseImpactParameter().error());
      refittedTrackDxyBSSigned.push_back(
          IPTools::signedTransverseImpactParameter(refittedTrack_i, electronRefTrackDir_i, beamSpotVertex)
              .second.value());
      refittedTrackDxyBSSignedErr.push_back(
          IPTools::signedTransverseImpactParameter(refittedTrack_i, electronRefTrackDir_i, beamSpotVertex)
              .second.error());
      refittedTrackIp3DBSSigned.push_back(
          IPTools::signedImpactParameter3D(refittedTrack_i, electronRefTrackDir_i, beamSpotVertex).second.value());
      refittedTrackIp3DBSSignedErr.push_back(
          IPTools::signedImpactParameter3D(refittedTrack_i, electronRefTrackDir_i, beamSpotVertex).second.error());
      refittedTrackIdx.push_back(refittedTrackIdx_counter);
      refittedTrackIdx1.push_back(refittedTrackIdx_counter);
      refittedTrackIdx_counter++;

      refittedTrackOriginalIdx.push_back(j);
      refittedTrackPt.push_back(refittedTrack_j.track().pt());
      refittedTrackPtErr.push_back(refittedTrack_j.track().ptError());
      refittedTrackPx.push_back(refittedTrack_j.track().px());
      refittedTrackPy.push_back(refittedTrack_j.track().py());
      refittedTrackPz.push_back(refittedTrack_j.track().pz());
      refittedTrackEta.push_back(refittedTrack_j.track().eta());
      refittedTrackEtaErr.push_back(refittedTrack_j.track().etaError());
      refittedTrackPhi.push_back(refittedTrack_j.track().phi());
      refittedTrackPhiErr.push_back(refittedTrack_j.track().phiError());
      refittedTrackCharge.push_back(refittedTrack_j.track().charge());
      refittedTrackNormChi2.push_back(refittedTrack_j.normalizedChi2());
      refittedTrackNdof.push_back(refittedTrack_j.ndof());
      refittedTrackChi2.push_back(refittedTrack_j.chi2());

      refittedTrackDzPV.push_back(refittedTrack_j.track().dz(pv.position()));
      refittedTrackDzPVErr.push_back(std::hypot(refittedTrack_j.track().dzError(), pv.zError()));
      TrajectoryStateClosestToPoint trajectoryPV_j = refittedTrack_j.trajectoryStateClosestToPoint(primaryVertex);
      refittedTrackDxyPVTraj.push_back(trajectoryPV_j.perigeeParameters().transverseImpactParameter());
      refittedTrackDxyPVTrajErr.push_back(trajectoryPV_j.perigeeError().transverseImpactParameterError());
      GlobalVector electronRefTrackDir_j(
          refittedTrack_j.track().px(), refittedTrack_j.track().py(), refittedTrack_j.track().pz());
      refittedTrackDxyPVSigned.push_back(
          IPTools::signedTransverseImpactParameter(refittedTrack_j, electronRefTrackDir_j, pv).second.value());
      refittedTrackDxyPVSignedErr.push_back(
          IPTools::signedTransverseImpactParameter(refittedTrack_j, electronRefTrackDir_j, pv).second.error());
      refittedTrackIp3DPVSigned.push_back(
          IPTools::signedImpactParameter3D(refittedTrack_j, electronRefTrackDir_j, pv).second.value());
      refittedTrackIp3DPVSignedErr.push_back(
          IPTools::signedImpactParameter3D(refittedTrack_j, electronRefTrackDir_j, pv).second.error());
      refittedTrackDxyBS.push_back(refittedTrack_j.track().dxy(bs));
      refittedTrackDxyBSErr.push_back(std::hypot(refittedTrack_j.track().dxyError(), beamSpotVertex.zError()));
      refittedTrackDzBS.push_back(refittedTrack_j.track().dz(bs));
      refittedTrackDzBSErr.push_back(std::hypot(refittedTrack_j.track().dzError(), beamSpotVertex.zError()));
      TrajectoryStateClosestToBeamLine trajectoryBS_j = refittedTrack_j.stateAtBeamLine();
      refittedTrackDxyBSTraj.push_back(trajectoryBS_j.transverseImpactParameter().value());
      refittedTrackDxyBSTrajErr.push_back(trajectoryBS_j.transverseImpactParameter().error());
      refittedTrackDxyBSSigned.push_back(
          IPTools::signedTransverseImpactParameter(refittedTrack_j, electronRefTrackDir_j, beamSpotVertex)
              .second.value());
      refittedTrackDxyBSSignedErr.push_back(
          IPTools::signedTransverseImpactParameter(refittedTrack_j, electronRefTrackDir_j, beamSpotVertex)
              .second.error());
      refittedTrackIp3DBSSigned.push_back(
          IPTools::signedImpactParameter3D(refittedTrack_j, electronRefTrackDir_j, beamSpotVertex).second.value());
      refittedTrackIp3DBSSignedErr.push_back(
          IPTools::signedImpactParameter3D(refittedTrack_j, electronRefTrackDir_j, beamSpotVertex).second.error());
      refittedTrackIdx.push_back(refittedTrackIdx_counter);
      refittedTrackIdx2.push_back(refittedTrackIdx_counter);
      refittedTrackIdx_counter++;

      // Perform kinematic fit to re-compute dielectron four-momentum (especially for better mass resolution)
      KinematicParticleFactoryFromTransientTrack pFactory;
      ParticleMass e_mass = 0.000511;
      float e_sigma = 0.00000001;
      float chi = 0.;
      float ndf = 0.;
      std::vector<RefCountedKinematicParticle> eleParticles;
      eleParticles.push_back(pFactory.particle(electronTransientTracks[0], e_mass, chi, ndf, e_sigma));
      eleParticles.push_back(pFactory.particle(electronTransientTracks[1], e_mass, chi, ndf, e_sigma));
      KinematicParticleVertexFitter fitter;
      try {
        RefCountedKinematicTree vertexFitTree = fitter.fit(eleParticles);
        if (vertexFitTree->isValid()) {
          vertexFitTree->movePointerToTheTop();
          auto diele_part = vertexFitTree->currentParticle();
          auto diele_state = diele_part->currentState();
          auto daughters = vertexFitTree->daughterParticles();
          refittedVertMass.push_back(diele_state.mass());
          refittedVertPt.push_back(diele_state.globalMomentum().transverse());
          refittedVertEta.push_back(diele_state.globalMomentum().eta());
          refittedVertPhi.push_back(diele_state.globalMomentum().phi());
        } else {
          refittedVertMass.push_back(-999.);
          refittedVertPt.push_back(-999.);
          refittedVertEta.push_back(-999.);
          refittedVertPhi.push_back(-999.);
        }
      } catch (std::exception& ex) {
        std::cout << "kinematic vertex fit failed!" << std::endl;
        refittedVertMass.push_back(-999.);
        refittedVertPt.push_back(-999.);
        refittedVertEta.push_back(-999.);
        refittedVertPhi.push_back(-999.);
      }
    }
  }

  auto vertexTab = std::make_unique<nanoaod::FlatTable>(nElectronVertices, "ElectronVertex", false, false);
  auto refittedTracksTab =
      std::make_unique<nanoaod::FlatTable>(nElectronVertices * 2, "ElectronVertexRefittedTracks", false, false);

  vertexTab->addColumn<float>("isValid", vertexIsValid, "");
  vertexTab->addColumn<float>("vxy", vxy, "");
  vertexTab->addColumn<float>("vxySigma", vxySigma, "");
  vertexTab->addColumn<float>("vxyz", vxyz, "");
  vertexTab->addColumn<float>("vxyzSigma", vxyzSigma, "");
  vertexTab->addColumn<float>("chi2", chi2, "");
  vertexTab->addColumn<float>("ndof", ndof, "");
  vertexTab->addColumn<float>("normChi2", normChi2, "");
  vertexTab->addColumn<float>("vx", vx, "");
  vertexTab->addColumn<float>("vy", vy, "");
  vertexTab->addColumn<float>("vz", vz, "");
  vertexTab->addColumn<float>("t", t, "");
  vertexTab->addColumn<float>("vxErr", vxErr, "");
  vertexTab->addColumn<float>("vyErr", vyErr, "");
  vertexTab->addColumn<float>("vzErr", vzErr, "");
  vertexTab->addColumn<float>("tErr", tErr, "");
  vertexTab->addColumn<float>("dR", dR, "");
  vertexTab->addColumn<float>("originalElectronIdx1", originalElectronIdx1, "");
  vertexTab->addColumn<float>("originalElectronIdx2", originalElectronIdx2, "");
  vertexTab->addColumn<float>("dca", DCA, "");
  vertexTab->addColumn<float>("dcaStatus", DCAstatus, "");
  vertexTab->addColumn<float>("dcax", DCAx, "");
  vertexTab->addColumn<float>("dcay", DCAy, "");
  vertexTab->addColumn<float>("dcaz", DCAz, "");
  vertexTab->addColumn<float>("hitsInFrontOfVert1", hitsInFrontOfVert1, "");
  vertexTab->addColumn<float>("hitsInFrontOfVert2", hitsInFrontOfVert2, "");
  vertexTab->addColumn<float>("missHitsAfterVert1", missHitsAfterVert1, "");
  vertexTab->addColumn<float>("missHitsAfterVert2", missHitsAfterVert2, "");
  vertexTab->addColumn<float>("refittedTrackIdx1", refittedTrackIdx1, "");
  vertexTab->addColumn<float>("refittedTrackIdx2", refittedTrackIdx2, "");
  vertexTab->addColumn<float>("massRefit", refittedVertMass, "");
  vertexTab->addColumn<float>("ptRefit", refittedVertPt, "");
  vertexTab->addColumn<float>("etaRefit", refittedVertEta, "");
  vertexTab->addColumn<float>("phiRefit", refittedVertPhi, "");

  iEvent.put(std::move(vertexTab), "ElectronVertex");

  refittedTracksTab->addColumn<float>("idx", refittedTrackIdx, "");
  refittedTracksTab->addColumn<float>("originalElectronIdx", refittedTrackOriginalIdx, "");
  refittedTracksTab->addColumn<float>("pt", refittedTrackPt, "");
  refittedTracksTab->addColumn<float>("ptErr", refittedTrackPtErr, "");
  refittedTracksTab->addColumn<float>("px", refittedTrackPx, "");
  refittedTracksTab->addColumn<float>("py", refittedTrackPy, "");
  refittedTracksTab->addColumn<float>("pz", refittedTrackPz, "");
  refittedTracksTab->addColumn<float>("eta", refittedTrackEta, "");
  refittedTracksTab->addColumn<float>("etaErr", refittedTrackEtaErr, "");
  refittedTracksTab->addColumn<float>("phi", refittedTrackPhi, "");
  refittedTracksTab->addColumn<float>("phiErr", refittedTrackPhiErr, "");
  refittedTracksTab->addColumn<float>("charge", refittedTrackCharge, "");
  refittedTracksTab->addColumn<float>("normChi2", refittedTrackNormChi2, "");
  refittedTracksTab->addColumn<float>("ndof", refittedTrackNdof, "");
  refittedTracksTab->addColumn<float>("chi2", refittedTrackChi2, "");
  refittedTracksTab->addColumn<float>("dzPV", refittedTrackDzPV, "");
  refittedTracksTab->addColumn<float>("dzPVErr", refittedTrackDzPVErr, "");
  refittedTracksTab->addColumn<float>("dxyPVTraj", refittedTrackDxyPVTraj, "");
  refittedTracksTab->addColumn<float>("dxyPVTrajErr", refittedTrackDxyPVTrajErr, "");
  refittedTracksTab->addColumn<float>("dxyPVSigned", refittedTrackDxyPVSigned, "");
  refittedTracksTab->addColumn<float>("dxyPVSignedErr", refittedTrackDxyPVSignedErr, "");
  refittedTracksTab->addColumn<float>("ip3DPVSigned", refittedTrackIp3DPVSigned, "");
  refittedTracksTab->addColumn<float>("ip3DPVSignedErr", refittedTrackIp3DPVSignedErr, "");
  refittedTracksTab->addColumn<float>("dxyBS", refittedTrackDxyBS, "");
  refittedTracksTab->addColumn<float>("dxyBSErr", refittedTrackDxyBSErr, "");
  refittedTracksTab->addColumn<float>("dzBS", refittedTrackDzBS, "");
  refittedTracksTab->addColumn<float>("dzBSErr", refittedTrackDzBSErr, "");
  refittedTracksTab->addColumn<float>("dxyBSTraj", refittedTrackDxyBSTraj, "");
  refittedTracksTab->addColumn<float>("dxyBSTrajErr", refittedTrackDxyBSTrajErr, "");
  refittedTracksTab->addColumn<float>("dxyBSSigned", refittedTrackDxyBSSigned, "");
  refittedTracksTab->addColumn<float>("dxyBSSignedErr", refittedTrackDxyBSSignedErr, "");
  refittedTracksTab->addColumn<float>("ip3DBSSigned", refittedTrackIp3DBSSigned, "");
  refittedTracksTab->addColumn<float>("ip3DBSSignedErr", refittedTrackIp3DBSSignedErr, "");

  iEvent.put(std::move(refittedTracksTab), "ElectronVertexRefittedTracks");
}

std::pair<float, float> ElectronVertexTableProducer::getVxy(const reco::Vertex electronVertex) const {
  float vxy = sqrt(electronVertex.x() * electronVertex.x() + electronVertex.y() * electronVertex.y());
  float vxySigma =
      (1 / vxy) * sqrt(electronVertex.x() * electronVertex.x() * electronVertex.xError() * electronVertex.xError() +
                       electronVertex.y() * electronVertex.y() * electronVertex.yError() * electronVertex.yError());
  return std::make_pair(vxy, vxySigma);
}

std::pair<float, float> ElectronVertexTableProducer::getVxyz(const reco::Vertex electronVertex) const {
  float vxyz = sqrt(electronVertex.x() * electronVertex.x() + electronVertex.y() * electronVertex.y() +
                    electronVertex.z() * electronVertex.z());
  float vxyzSigma =
      (1 / vxyz) * sqrt(electronVertex.x() * electronVertex.x() * electronVertex.xError() * electronVertex.xError() +
                        electronVertex.y() * electronVertex.y() * electronVertex.yError() * electronVertex.yError() +
                        electronVertex.z() * electronVertex.z() * electronVertex.zError() * electronVertex.zError());
  return std::make_pair(vxyz, vxyzSigma);
}

/**
*  Proximity between the electrons based on EXO-23-010
*  Getting Distance of Closest Approach between electron tracks using TwoTrackMinimumDistance
*  Returns tuple of distance (float), error of distance (float) and crossing point (GlobalPoint)
**/
std::tuple<float, float, GlobalPoint> ElectronVertexTableProducer::getDistanceBetweenElectronTracks(
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
DEFINE_FWK_MODULE(ElectronVertexTableProducer);
