// System include files
#include <cmath>
#include <memory>

#include <map>
#include <string>
#include <vector>

// root objects
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TDirectory.h"
#include "TTree.h"

#include <Math/GenVector/VectorUtil.h>

#include "Calibration/IsolatedParticles/interface/FindCaloHit.h"
#include "Calibration/IsolatedParticles/interface/eECALMatrix.h"
#include "Calibration/IsolatedParticles/interface/eHCALMatrix.h"
#include "Calibration/IsolatedParticles/interface/MatchingSimTrack.h"
#include "Calibration/IsolatedParticles/interface/CaloSimInfo.h"
#include "Calibration/IsolatedParticles/interface/TrackSelection.h"
#include "Calibration/IsolatedParticles/interface/CaloPropagateTrack.h"
#include "Calibration/IsolatedParticles/interface/ChargeIsolation.h"
#include "Calibration/IsolatedParticles/interface/eCone.h"
#include "Calibration/IsolatedParticles/interface/eHCALMatrixExtra.h"

#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Candidate/interface/Candidate.h"

// muons and tracks
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
// Vertices
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
// Calorimeters
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"

// Jets in the event
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/JetExtendedAssociation.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

// TFile Service
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

// SimHit
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

// track associator
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"

class IsolatedTracksHcalScale : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit IsolatedTracksHcalScale(const edm::ParameterSet &);
  ~IsolatedTracksHcalScale() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void endJob() override;

  void clearTreeVectors();

private:
  bool doMC_;
  int myverbose_;
  std::string theTrackQuality_, minQuality_;
  spr::trackSelectionParameters selectionParameters_;
  double a_mipR_, a_coneR_, a_charIsoR_, a_neutIsoR_;
  double tMinE_, tMaxE_;

  TrackerHitAssociator::Config trackerHitAssociatorConfig_;

  edm::EDGetTokenT<reco::TrackCollection> tok_genTrack_;
  edm::EDGetTokenT<reco::VertexCollection> tok_recVtx_;
  edm::EDGetTokenT<reco::BeamSpot> tok_bs_;
  edm::EDGetTokenT<EcalRecHitCollection> tok_EB_;
  edm::EDGetTokenT<EcalRecHitCollection> tok_EE_;
  edm::EDGetTokenT<HBHERecHitCollection> tok_hbhe_;

  edm::EDGetTokenT<edm::SimTrackContainer> tok_simTk_;
  edm::EDGetTokenT<edm::SimVertexContainer> tok_simVtx_;
  edm::EDGetTokenT<edm::PCaloHitContainer> tok_caloEB_;
  edm::EDGetTokenT<edm::PCaloHitContainer> tok_caloEE_;
  edm::EDGetTokenT<edm::PCaloHitContainer> tok_caloHH_;

  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom_;
  edm::ESGetToken<CaloTopology, CaloTopologyRecord> tok_caloTopology_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> tok_magField_;
  edm::ESGetToken<EcalChannelStatus, EcalChannelStatusRcd> tok_ecalChStatus_;
  edm::ESGetToken<EcalSeverityLevelAlgo, EcalSeverityLevelAlgoRcd> tok_sevlv_;

  int nEventProc_;

  TTree *tree_;

  int t_nTracks, t_RunNo, t_EvtNo, t_Lumi, t_Bunch;
  std::vector<double> *t_trackP, *t_trackPt, *t_trackEta, *t_trackPhi;
  std::vector<double> *t_trackHcalEta, *t_trackHcalPhi, *t_eHCALDR;
  std::vector<double> *t_hCone, *t_conehmaxNearP, *t_eMipDR, *t_eECALDR;
  std::vector<double> *t_e11x11_20Sig, *t_e15x15_20Sig;
  std::vector<double> *t_eMipDR_1, *t_eECALDR_1, *t_eMipDR_2, *t_eECALDR_2;
  std::vector<double> *t_hConeHB, *t_eHCALDRHB;
  std::vector<double> *t_hsimInfoMatched, *t_hsimInfoRest, *t_hsimInfoPhoton;
  std::vector<double> *t_hsimInfoNeutHad, *t_hsimInfoCharHad, *t_hsimInfoPdgMatched;
  std::vector<double> *t_hsimInfoTotal, *t_hsim;
  std::vector<int> *t_hsimInfoNMatched, *t_hsimInfoNTotal, *t_hsimInfoNNeutHad;
  std::vector<int> *t_hsimInfoNCharHad, *t_hsimInfoNPhoton, *t_hsimInfoNRest;
  std::vector<int> *t_nSimHits;
};

IsolatedTracksHcalScale::IsolatedTracksHcalScale(const edm::ParameterSet &iConfig)
    : doMC_(iConfig.getUntrackedParameter<bool>("DoMC", false)),
      myverbose_(iConfig.getUntrackedParameter<int>("Verbosity", 5)),
      theTrackQuality_(iConfig.getUntrackedParameter<std::string>("TrackQuality", "highPurity")),
      a_mipR_(iConfig.getUntrackedParameter<double>("ConeRadiusMIP", 14.0)),
      a_coneR_(iConfig.getUntrackedParameter<double>("ConeRadius", 34.98)),
      tMinE_(iConfig.getUntrackedParameter<double>("TimeMinCutECAL", -500.)),
      tMaxE_(iConfig.getUntrackedParameter<double>("TimeMaxCutECAL", 500.)),
      trackerHitAssociatorConfig_(consumesCollector()) {
  usesResource(TFileService::kSharedResource);

  //now do what ever initialization is needed
  reco::TrackBase::TrackQuality trackQuality = reco::TrackBase::qualityByName(theTrackQuality_);
  selectionParameters_.minPt = iConfig.getUntrackedParameter<double>("MinTrackPt", 10.0);
  selectionParameters_.minQuality = trackQuality;
  selectionParameters_.maxDxyPV = iConfig.getUntrackedParameter<double>("MaxDxyPV", 0.2);
  selectionParameters_.maxDzPV = iConfig.getUntrackedParameter<double>("MaxDzPV", 5.0);
  selectionParameters_.maxChi2 = iConfig.getUntrackedParameter<double>("MaxChi2", 5.0);
  selectionParameters_.maxDpOverP = iConfig.getUntrackedParameter<double>("MaxDpOverP", 0.1);
  selectionParameters_.minOuterHit = iConfig.getUntrackedParameter<int>("MinOuterHit", 4);
  selectionParameters_.minLayerCrossed = iConfig.getUntrackedParameter<int>("MinLayerCrossed", 8);
  selectionParameters_.maxInMiss = iConfig.getUntrackedParameter<int>("MaxInMiss", 0);
  selectionParameters_.maxOutMiss = iConfig.getUntrackedParameter<int>("MaxOutMiss", 0);
  a_charIsoR_ = a_coneR_ + 28.9;
  a_neutIsoR_ = a_charIsoR_ * 0.726;

  tok_genTrack_ = consumes<reco::TrackCollection>(edm::InputTag("generalTracks"));
  tok_recVtx_ = consumes<reco::VertexCollection>(edm::InputTag("offlinePrimaryVertices"));
  tok_bs_ = consumes<reco::BeamSpot>(edm::InputTag("offlineBeamSpot"));
  tok_EB_ = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit", "EcalRecHitsEB"));
  tok_EE_ = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit", "EcalRecHitsEE"));
  tok_hbhe_ = consumes<HBHERecHitCollection>(edm::InputTag("hbhereco"));
  tok_simTk_ = consumes<edm::SimTrackContainer>(edm::InputTag("g4SimHits"));
  tok_simVtx_ = consumes<edm::SimVertexContainer>(edm::InputTag("g4SimHits"));
  tok_caloEB_ = consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits", "EcalHitsEB"));
  tok_caloEE_ = consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits", "EcalHitsEE"));
  tok_caloHH_ = consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits", "HcalHits"));

  if (myverbose_ >= 0) {
    edm::LogVerbatim("IsoTrack") << "Parameters read from config file \n"
                                 << " doMC " << doMC_ << "\t myverbose " << myverbose_ << "\t minPt "
                                 << selectionParameters_.minPt << "\t theTrackQuality " << theTrackQuality_
                                 << "\t minQuality " << selectionParameters_.minQuality << "\t maxDxyPV "
                                 << selectionParameters_.maxDxyPV << "\t maxDzPV " << selectionParameters_.maxDzPV
                                 << "\t maxChi2 " << selectionParameters_.maxChi2 << "\t maxDpOverP "
                                 << selectionParameters_.maxDpOverP << "\t minOuterHit "
                                 << selectionParameters_.minOuterHit << "\t minLayerCrossed "
                                 << selectionParameters_.minLayerCrossed << "\t maxInMiss "
                                 << selectionParameters_.maxInMiss << "\t maxOutMiss "
                                 << selectionParameters_.maxOutMiss << "\t a_coneR " << a_coneR_ << "\t a_charIsoR "
                                 << a_charIsoR_ << "\t a_neutIsoR " << a_neutIsoR_ << "\t a_mipR " << a_mipR_
                                 << "\t time Range (" << tMinE_ << ":" << tMaxE_ << ")";
  }

  tok_geom_ = esConsumes<CaloGeometry, CaloGeometryRecord>();
  tok_caloTopology_ = esConsumes<CaloTopology, CaloTopologyRecord>();
  tok_magField_ = esConsumes<MagneticField, IdealMagneticFieldRecord>();
  tok_ecalChStatus_ = esConsumes<EcalChannelStatus, EcalChannelStatusRcd>();
  tok_sevlv_ = esConsumes<EcalSeverityLevelAlgo, EcalSeverityLevelAlgoRcd>();
}

void IsolatedTracksHcalScale::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<bool>("doMC", false);
  desc.addUntracked<int>("Verbosity", 0);
  desc.addUntracked<std::string>("TrackQuality", "highPurity");
  desc.addUntracked<double>("MinTrackPt", 10.0);
  desc.addUntracked<double>("MaxDxyPV", 0.02);
  desc.addUntracked<double>("MaxDzPV", 0.02);
  desc.addUntracked<double>("MaxChi2", 5.0);
  desc.addUntracked<double>("MaxDpOverP", 0.1);
  desc.addUntracked<int>("MinOuterHit", 4);
  desc.addUntracked<int>("MinLayerCrossed", 8);
  desc.addUntracked<int>("MaxInMiss", 0);
  desc.addUntracked<int>("MaxOutMiss", 0);
  desc.addUntracked<double>("ConeRadius", 34.98);
  desc.addUntracked<double>("ConeRadiusMIP", 14.0);
  desc.addUntracked<double>("TimeMinCutECAL", -500.0);
  desc.addUntracked<double>("TimeMaxCutECAL", 500.0);
  descriptions.add("isolatedTracksHcalScale", desc);
}

void IsolatedTracksHcalScale::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  const CaloGeometry *geo = &iSetup.getData(tok_geom_);
  const MagneticField *bField = &iSetup.getData(tok_magField_);
  const EcalChannelStatus *theEcalChStatus = &iSetup.getData(tok_ecalChStatus_);
  const EcalSeverityLevelAlgo *sevlv = &iSetup.getData(tok_sevlv_);
  const CaloTopology *caloTopology = &iSetup.getData(tok_caloTopology_);

  clearTreeVectors();

  ++nEventProc_;

  t_RunNo = iEvent.id().run();
  t_EvtNo = iEvent.id().event();
  t_Lumi = iEvent.luminosityBlock();
  t_Bunch = iEvent.bunchCrossing();
  if (myverbose_ > 0)
    edm::LogVerbatim("IsoTrack") << nEventProc_ << " Run " << t_RunNo << " Event " << t_EvtNo << " Lumi " << t_Lumi
                                 << " Bunch " << t_Bunch;

  edm::Handle<reco::TrackCollection> trkCollection;
  iEvent.getByToken(tok_genTrack_, trkCollection);

  edm::Handle<reco::VertexCollection> recVtxs;
  iEvent.getByToken(tok_recVtx_, recVtxs);

  // Get the beamspot
  edm::Handle<reco::BeamSpot> beamSpotH;
  iEvent.getByToken(tok_bs_, beamSpotH);

  math::XYZPoint leadPV(0, 0, 0);
  if (!recVtxs->empty() && !((*recVtxs)[0].isFake())) {
    leadPV = math::XYZPoint((*recVtxs)[0].x(), (*recVtxs)[0].y(), (*recVtxs)[0].z());
  } else if (beamSpotH.isValid()) {
    leadPV = beamSpotH->position();
  }

  if (myverbose_ > 0) {
    edm::LogVerbatim("IsoTrack") << "Primary Vertex " << leadPV;
    if (beamSpotH.isValid())
      edm::LogVerbatim("IsoTrack") << "Beam Spot " << beamSpotH->position();
  }

  std::vector<spr::propagatedTrackDirection> trkCaloDirections;
  spr::propagateCALO(trkCollection, geo, bField, theTrackQuality_, trkCaloDirections, (myverbose_ > 2));
  std::vector<spr::propagatedTrackDirection>::const_iterator trkDetItr;

  edm::Handle<EcalRecHitCollection> barrelRecHitsHandle;
  edm::Handle<EcalRecHitCollection> endcapRecHitsHandle;
  iEvent.getByToken(tok_EB_, barrelRecHitsHandle);
  iEvent.getByToken(tok_EE_, endcapRecHitsHandle);

  edm::Handle<HBHERecHitCollection> hbhe;
  iEvent.getByToken(tok_hbhe_, hbhe);
  const HBHERecHitCollection Hithbhe = *(hbhe.product());

  //get Handles to SimTracks and SimHits
  edm::Handle<edm::SimTrackContainer> SimTk;
  edm::Handle<edm::SimVertexContainer> SimVtx;

  //get Handles to PCaloHitContainers of eb/ee/hbhe
  edm::Handle<edm::PCaloHitContainer> pcaloeb;
  edm::Handle<edm::PCaloHitContainer> pcaloee;
  edm::Handle<edm::PCaloHitContainer> pcalohh;

  //associates tracker rechits/simhits to a track
  std::unique_ptr<TrackerHitAssociator> associate;

  if (doMC_) {
    iEvent.getByToken(tok_simTk_, SimTk);
    iEvent.getByToken(tok_simVtx_, SimVtx);
    iEvent.getByToken(tok_caloEB_, pcaloeb);
    iEvent.getByToken(tok_caloEE_, pcaloee);
    iEvent.getByToken(tok_caloHH_, pcalohh);
    associate = std::make_unique<TrackerHitAssociator>(iEvent, trackerHitAssociatorConfig_);
  }

  unsigned int nTracks = 0;
  for (trkDetItr = trkCaloDirections.begin(), nTracks = 0; trkDetItr != trkCaloDirections.end();
       trkDetItr++, nTracks++) {
    const reco::Track *pTrack = &(*(trkDetItr->trkItr));
    if (spr::goodTrack(pTrack, leadPV, selectionParameters_, (myverbose_ > 2)) && trkDetItr->okECAL &&
        trkDetItr->okHCAL) {
      int nRH_eMipDR = 0, nRH_eDR = 0, nNearTRKs = 0, nRecHitsCone = -99;
      double distFromHotCell = -99.0, distFromHotCell2 = -99.0;
      int ietaHotCell = -99, iphiHotCell = -99;
      int ietaHotCell2 = -99, iphiHotCell2 = -99;
      GlobalPoint gposHotCell(0., 0., 0.), gposHotCell2(0., 0., 0.);
      std::vector<DetId> coneRecHitDetIds, coneRecHitDetIds2;
      std::pair<double, bool> e11x11_20SigP, e15x15_20SigP;
      double hCone = spr::eCone_hcal(geo,
                                     hbhe,
                                     trkDetItr->pointHCAL,
                                     trkDetItr->pointECAL,
                                     a_coneR_,
                                     trkDetItr->directionHCAL,
                                     nRecHitsCone,
                                     coneRecHitDetIds,
                                     distFromHotCell,
                                     ietaHotCell,
                                     iphiHotCell,
                                     gposHotCell,
                                     -1);
      double hConeHB = spr::eCone_hcal(geo,
                                       hbhe,
                                       trkDetItr->pointHCAL,
                                       trkDetItr->pointECAL,
                                       a_coneR_,
                                       trkDetItr->directionHCAL,
                                       nRecHitsCone,
                                       coneRecHitDetIds,
                                       distFromHotCell,
                                       ietaHotCell,
                                       iphiHotCell,
                                       gposHotCell,
                                       (int)(HcalBarrel));
      double eHCALDR = spr::eCone_hcal(geo,
                                       hbhe,
                                       trkDetItr->pointHCAL,
                                       trkDetItr->pointECAL,
                                       a_charIsoR_,
                                       trkDetItr->directionHCAL,
                                       nRecHitsCone,
                                       coneRecHitDetIds2,
                                       distFromHotCell2,
                                       ietaHotCell2,
                                       iphiHotCell2,
                                       gposHotCell2,
                                       -1);
      double eHCALDRHB = spr::eCone_hcal(geo,
                                         hbhe,
                                         trkDetItr->pointHCAL,
                                         trkDetItr->pointECAL,
                                         a_charIsoR_,
                                         trkDetItr->directionHCAL,
                                         nRecHitsCone,
                                         coneRecHitDetIds2,
                                         distFromHotCell2,
                                         ietaHotCell2,
                                         iphiHotCell2,
                                         gposHotCell2,
                                         (int)(HcalBarrel));

      double conehmaxNearP =
          spr::chargeIsolationCone(nTracks, trkCaloDirections, a_charIsoR_, nNearTRKs, (myverbose_ > 3));

      double eMipDR = spr::eCone_ecal(geo,
                                      barrelRecHitsHandle,
                                      endcapRecHitsHandle,
                                      trkDetItr->pointHCAL,
                                      trkDetItr->pointECAL,
                                      a_mipR_,
                                      trkDetItr->directionECAL,
                                      nRH_eMipDR);
      double eECALDR = spr::eCone_ecal(geo,
                                       barrelRecHitsHandle,
                                       endcapRecHitsHandle,
                                       trkDetItr->pointHCAL,
                                       trkDetItr->pointECAL,
                                       a_neutIsoR_,
                                       trkDetItr->directionECAL,
                                       nRH_eDR);
      double eMipDR_1 = spr::eCone_ecal(geo,
                                        barrelRecHitsHandle,
                                        endcapRecHitsHandle,
                                        trkDetItr->pointHCAL,
                                        trkDetItr->pointECAL,
                                        a_mipR_,
                                        trkDetItr->directionECAL,
                                        nRH_eMipDR,
                                        0.030,
                                        0.150);
      double eECALDR_1 = spr::eCone_ecal(geo,
                                         barrelRecHitsHandle,
                                         endcapRecHitsHandle,
                                         trkDetItr->pointHCAL,
                                         trkDetItr->pointECAL,
                                         a_neutIsoR_,
                                         trkDetItr->directionECAL,
                                         nRH_eDR,
                                         0.030,
                                         0.150);
      double eMipDR_2 = spr::eCone_ecal(geo,
                                        barrelRecHitsHandle,
                                        endcapRecHitsHandle,
                                        trkDetItr->pointHCAL,
                                        trkDetItr->pointECAL,
                                        a_mipR_,
                                        trkDetItr->directionECAL,
                                        nRH_eMipDR,
                                        0.060,
                                        0.300);
      double eECALDR_2 = spr::eCone_ecal(geo,
                                         barrelRecHitsHandle,
                                         endcapRecHitsHandle,
                                         trkDetItr->pointHCAL,
                                         trkDetItr->pointECAL,
                                         a_neutIsoR_,
                                         trkDetItr->directionECAL,
                                         nRH_eDR,
                                         0.060,
                                         0.300);

      HcalDetId closestCell = (HcalDetId)(trkDetItr->detIdHCAL);

      e11x11_20SigP = spr::eECALmatrix(trkDetItr->detIdECAL,
                                       barrelRecHitsHandle,
                                       endcapRecHitsHandle,
                                       *theEcalChStatus,
                                       geo,
                                       caloTopology,
                                       sevlv,
                                       5,
                                       5,
                                       0.060,
                                       0.300,
                                       tMinE_,
                                       tMaxE_);
      e15x15_20SigP = spr::eECALmatrix(trkDetItr->detIdECAL,
                                       barrelRecHitsHandle,
                                       endcapRecHitsHandle,
                                       *theEcalChStatus,
                                       geo,
                                       caloTopology,
                                       sevlv,
                                       7,
                                       7,
                                       0.060,
                                       0.300,
                                       tMinE_,
                                       tMaxE_);

      // Fill the tree Branches here
      t_trackP->push_back(pTrack->p());
      t_trackPt->push_back(pTrack->pt());
      t_trackEta->push_back(pTrack->momentum().eta());
      t_trackPhi->push_back(pTrack->momentum().phi());
      t_trackHcalEta->push_back(closestCell.ieta());
      t_trackHcalPhi->push_back(closestCell.iphi());
      t_hCone->push_back(hCone);
      t_conehmaxNearP->push_back(conehmaxNearP);
      t_eMipDR->push_back(eMipDR);
      t_eECALDR->push_back(eECALDR);
      t_eHCALDR->push_back(eHCALDR);
      t_e11x11_20Sig->push_back(e11x11_20SigP.first);
      t_e15x15_20Sig->push_back(e15x15_20SigP.first);
      t_eMipDR_1->push_back(eMipDR_1);
      t_eECALDR_1->push_back(eECALDR_1);
      t_eMipDR_2->push_back(eMipDR_2);
      t_eECALDR_2->push_back(eECALDR_2);
      t_hConeHB->push_back(hConeHB);
      t_eHCALDRHB->push_back(eHCALDRHB);

      if (myverbose_ > 0) {
        edm::LogVerbatim("IsoTrack") << "Track p " << pTrack->p() << " pt " << pTrack->pt() << " eta "
                                     << pTrack->momentum().eta() << " phi " << pTrack->momentum().phi()
                                     << " ieta/iphi (" << closestCell.ieta() << ", " << closestCell.iphi()
                                     << ") Energy in cone " << hCone << " Charge Isolation " << conehmaxNearP
                                     << " eMIP (" << eMipDR << ", " << eMipDR_1 << ", " << eMipDR_2 << ")"
                                     << " Neutral isolation (ECAL) (" << eECALDR - eMipDR << ", "
                                     << eECALDR_1 - eMipDR_1 << ", " << eECALDR_2 - eMipDR_2 << ") (ECAL NxN) "
                                     << e15x15_20SigP.first - e11x11_20SigP.first << " (HCAL) " << eHCALDR - hCone;
      }

      if (doMC_) {
        int nSimHits = -999;
        double hsim;
        std::map<std::string, double> hsimInfo;
        std::vector<int> multiplicity;
        hsim = spr::eCone_hcal(
            geo, pcalohh, trkDetItr->pointHCAL, trkDetItr->pointECAL, a_coneR_, trkDetItr->directionHCAL, nSimHits);
        hsimInfo = spr::eHCALSimInfoCone(iEvent,
                                         pcalohh,
                                         SimTk,
                                         SimVtx,
                                         pTrack,
                                         *associate,
                                         geo,
                                         trkDetItr->pointHCAL,
                                         trkDetItr->pointECAL,
                                         a_coneR_,
                                         trkDetItr->directionHCAL,
                                         multiplicity);

        t_hsimInfoMatched->push_back(hsimInfo["eMatched"]);
        t_hsimInfoRest->push_back(hsimInfo["eRest"]);
        t_hsimInfoPhoton->push_back(hsimInfo["eGamma"]);
        t_hsimInfoNeutHad->push_back(hsimInfo["eNeutralHad"]);
        t_hsimInfoCharHad->push_back(hsimInfo["eChargedHad"]);
        t_hsimInfoPdgMatched->push_back(hsimInfo["pdgMatched"]);
        t_hsimInfoTotal->push_back(hsimInfo["eTotal"]);

        t_hsimInfoNMatched->push_back(multiplicity.at(0));
        t_hsimInfoNTotal->push_back(multiplicity.at(1));
        t_hsimInfoNNeutHad->push_back(multiplicity.at(2));
        t_hsimInfoNCharHad->push_back(multiplicity.at(3));
        t_hsimInfoNPhoton->push_back(multiplicity.at(4));
        t_hsimInfoNRest->push_back(multiplicity.at(5));

        t_hsim->push_back(hsim);
        t_nSimHits->push_back(nSimHits);

        if (myverbose_ > 0) {
          edm::LogVerbatim("IsoTrack") << "Matched (E) " << hsimInfo["eMatched"] << " (N) " << multiplicity.at(0)
                                       << " Rest (E) " << hsimInfo["eRest"] << " (N) " << multiplicity.at(1)
                                       << " Gamma (E) " << hsimInfo["eGamma"] << " (N) " << multiplicity.at(2)
                                       << " Neutral Had (E) " << hsimInfo["eNeutralHad"] << " (N) "
                                       << multiplicity.at(3) << " Charged Had (E) " << hsimInfo["eChargedHad"]
                                       << " (N) " << multiplicity.at(4) << " Total (E) " << hsimInfo["eTotal"]
                                       << " (N) " << multiplicity.at(5) << " PDG " << hsimInfo["pdgMatched"]
                                       << " Total E " << hsim << " NHit " << nSimHits;
        }
      }
    }
  }

  tree_->Fill();
}

void IsolatedTracksHcalScale::beginJob() {
  nEventProc_ = 0;
  edm::Service<TFileService> fs;

  //////Book Tree
  tree_ = fs->make<TTree>("tree", "tree");
  tree_->SetAutoSave(10000);

  tree_->Branch("t_RunNo", &t_RunNo, "t_RunNo/I");
  tree_->Branch("t_Lumi", &t_Lumi, "t_Lumi/I");
  tree_->Branch("t_Bunch", &t_Bunch, "t_Bunch/I");

  t_trackP = new std::vector<double>();
  t_trackPt = new std::vector<double>();
  t_trackEta = new std::vector<double>();
  t_trackPhi = new std::vector<double>();
  t_trackHcalEta = new std::vector<double>();
  t_trackHcalPhi = new std::vector<double>();
  t_hCone = new std::vector<double>();
  t_conehmaxNearP = new std::vector<double>();
  t_eMipDR = new std::vector<double>();
  t_eECALDR = new std::vector<double>();
  t_eHCALDR = new std::vector<double>();
  t_e11x11_20Sig = new std::vector<double>();
  t_e15x15_20Sig = new std::vector<double>();
  t_eMipDR_1 = new std::vector<double>();
  t_eECALDR_1 = new std::vector<double>();
  t_eMipDR_2 = new std::vector<double>();
  t_eECALDR_2 = new std::vector<double>();
  t_hConeHB = new std::vector<double>();
  t_eHCALDRHB = new std::vector<double>();

  tree_->Branch("t_trackP", "std::vector<double>", &t_trackP);
  tree_->Branch("t_trackPt", "std::vector<double>", &t_trackPt);
  tree_->Branch("t_trackEta", "std::vector<double>", &t_trackEta);
  tree_->Branch("t_trackPhi", "std::vector<double>", &t_trackPhi);
  tree_->Branch("t_trackHcalEta", "std::vector<double>", &t_trackHcalEta);
  tree_->Branch("t_trackHcalPhi", "std::vector<double>", &t_trackHcalPhi);
  tree_->Branch("t_hCone", "std::vector<double>", &t_hCone);
  tree_->Branch("t_conehmaxNearP", "std::vector<double>", &t_conehmaxNearP);
  tree_->Branch("t_eMipDR", "std::vector<double>", &t_eMipDR);
  tree_->Branch("t_eECALDR", "std::vector<double>", &t_eECALDR);
  tree_->Branch("t_eHCALDR", "std::vector<double>", &t_eHCALDR);
  tree_->Branch("t_e11x11_20Sig", "std::vector<double>", &t_e11x11_20Sig);
  tree_->Branch("t_e15x15_20Sig", "std::vector<double>", &t_e15x15_20Sig);
  tree_->Branch("t_eMipDR_1", "std::vector<double>", &t_eMipDR_1);
  tree_->Branch("t_eECALDR_1", "std::vector<double>", &t_eECALDR_1);
  tree_->Branch("t_eMipDR_2", "std::vector<double>", &t_eMipDR_2);
  tree_->Branch("t_eECALDR_2", "std::vector<double>", &t_eECALDR_2);
  tree_->Branch("t_hConeHB", "std::vector<double>", &t_hConeHB);
  tree_->Branch("t_eHCALDRHB", "std::vector<double>", &t_eHCALDRHB);

  if (doMC_) {
    t_hsimInfoMatched = new std::vector<double>();
    t_hsimInfoRest = new std::vector<double>();
    t_hsimInfoPhoton = new std::vector<double>();
    t_hsimInfoNeutHad = new std::vector<double>();
    t_hsimInfoCharHad = new std::vector<double>();
    t_hsimInfoPdgMatched = new std::vector<double>();
    t_hsimInfoTotal = new std::vector<double>();
    t_hsimInfoNMatched = new std::vector<int>();
    t_hsimInfoNTotal = new std::vector<int>();
    t_hsimInfoNNeutHad = new std::vector<int>();
    t_hsimInfoNCharHad = new std::vector<int>();
    t_hsimInfoNPhoton = new std::vector<int>();
    t_hsimInfoNRest = new std::vector<int>();
    t_hsim = new std::vector<double>();
    t_nSimHits = new std::vector<int>();

    tree_->Branch("t_hsimInfoMatched", "std::vector<double>", &t_hsimInfoMatched);
    tree_->Branch("t_hsimInfoRest", "std::vector<double>", &t_hsimInfoRest);
    tree_->Branch("t_hsimInfoPhoton", "std::vector<double>", &t_hsimInfoPhoton);
    tree_->Branch("t_hsimInfoNeutHad", "std::vector<double>", &t_hsimInfoNeutHad);
    tree_->Branch("t_hsimInfoCharHad", "std::vector<double>", &t_hsimInfoCharHad);
    tree_->Branch("t_hsimInfoPdgMatched", "std::vector<double>", &t_hsimInfoPdgMatched);
    tree_->Branch("t_hsimInfoTotal", "std::vector<double>", &t_hsimInfoTotal);
    tree_->Branch("t_hsimInfoNMatched", "std::vector<int>", &t_hsimInfoNMatched);
    tree_->Branch("t_hsimInfoNTotal", "std::vector<int>", &t_hsimInfoNTotal);
    tree_->Branch("t_hsimInfoNNeutHad", "std::vector<int>", &t_hsimInfoNNeutHad);
    tree_->Branch("t_hsimInfoNCharHad", "std::vector<int>", &t_hsimInfoNCharHad);
    tree_->Branch("t_hsimInfoNPhoton", "std::vector<int>", &t_hsimInfoNPhoton);
    tree_->Branch("t_hsimInfoNRest", "std::vector<int>", &t_hsimInfoNRest);
    tree_->Branch("t_hsim", "std::vector<double>", &t_hsim);
    tree_->Branch("t_nSimHits", "std::vector<int>", &t_nSimHits);
  }
}

void IsolatedTracksHcalScale::endJob() { edm::LogVerbatim("IsoTrack") << "Number of Events Processed " << nEventProc_; }

void IsolatedTracksHcalScale::clearTreeVectors() {
  t_trackP->clear();
  t_trackPt->clear();
  t_trackEta->clear();
  t_trackPhi->clear();
  t_trackHcalEta->clear();
  t_trackHcalPhi->clear();
  t_hCone->clear();
  t_conehmaxNearP->clear();
  t_eMipDR->clear();
  t_eECALDR->clear();
  t_eHCALDR->clear();
  t_e11x11_20Sig->clear();
  t_e15x15_20Sig->clear();
  t_eMipDR_1->clear();
  t_eECALDR_1->clear();
  t_eMipDR_2->clear();
  t_eECALDR_2->clear();
  t_hConeHB->clear();
  t_eHCALDRHB->clear();

  if (doMC_) {
    t_hsimInfoMatched->clear();
    t_hsimInfoRest->clear();
    t_hsimInfoPhoton->clear();
    t_hsimInfoNeutHad->clear();
    t_hsimInfoCharHad->clear();
    t_hsimInfoPdgMatched->clear();
    t_hsimInfoTotal->clear();
    t_hsimInfoNMatched->clear();
    t_hsimInfoNTotal->clear();
    t_hsimInfoNNeutHad->clear();
    t_hsimInfoNCharHad->clear();
    t_hsimInfoNPhoton->clear();
    t_hsimInfoNRest->clear();
    t_hsim->clear();
    t_nSimHits->clear();
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(IsolatedTracksHcalScale);
