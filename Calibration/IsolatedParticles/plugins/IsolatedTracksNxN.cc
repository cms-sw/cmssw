// -*- C++ -*
//
// Package:    IsolatedParticles
// Class:      IsolatedTracksNxN
//
/**\class IsolatedTracksNxN IsolatedTracksNxN.cc Calibration/IsolatedParticles/plugins/IsolatedTracksNxN.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Seema Sharma
//         Created:  Mon Aug 10 15:30:40 CST 2009
//
//

// system include files
#include <cmath>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// user include files
#include <Math/GenVector/VectorUtil.h>

// root objects
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TDirectory.h"
#include "TTree.h"

#include "Calibration/IsolatedParticles/interface/CaloSimInfo.h"
#include "Calibration/IsolatedParticles/interface/CaloPropagateTrack.h"
#include "Calibration/IsolatedParticles/interface/ChargeIsolation.h"
#include "Calibration/IsolatedParticles/interface/eECALMatrix.h"
#include "Calibration/IsolatedParticles/interface/eHCALMatrix.h"
#include "Calibration/IsolatedParticles/interface/eHCALMatrixExtra.h"
#include "Calibration/IsolatedParticles/interface/FindCaloHit.h"
#include "Calibration/IsolatedParticles/interface/MatchingSimTrack.h"

#include "CondFormats/DataRecord/interface/EcalChannelStatusRcd.h"
//L1 trigger Menus etc
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskAlgoTrigRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskTechTrigRcd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Candidate/interface/Candidate.h"
// muons and tracks
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackBase.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
// Vertices
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
// Calorimeters
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
// Trigger
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"
//L1 objects
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
// Jets in the event
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/JetExtendedAssociation.h"

#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
// TFile Service
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

// ecal / hcal
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgoRcd.h"

// SimHit + SimTrack
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

// track associator
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"

class IsolatedTracksNxN : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit IsolatedTracksNxN(const edm::ParameterSet &);
  ~IsolatedTracksNxN() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void endJob() override;

  void printTrack(const reco::Track *pTrack);

  void bookHistograms();

  void clearTreeVectors();

private:
  std::unique_ptr<L1GtUtils> m_l1GtUtils;
  static constexpr size_t nL1BitsMax = 128;
  TrackerHitAssociator::Config trackerHitAssociatorConfig_;

  // map of trig bit, algo name and num events passed
  std::map<std::pair<unsigned int, std::string>, int> l1AlgoMap_;
  std::vector<unsigned int> m_triggerMaskAlgoTrig;

  const bool doMC_, writeAllTracks_;
  const int myverbose_;
  const double pvTracksPtMin_;
  const int debugTrks_;
  const bool printTrkHitPattern_;
  const double minTrackP_, maxTrackEta_;
  const bool debugL1Info_, L1TriggerAlgoInfo_;
  const double tMinE_, tMaxE_, tMinH_, tMaxH_;
  bool initL1_;
  int nEventProc_, nbad_;

  edm::EDGetTokenT<l1extra::L1JetParticleCollection> tok_L1extTauJet_;
  edm::EDGetTokenT<l1extra::L1JetParticleCollection> tok_L1extCenJet_;
  edm::EDGetTokenT<l1extra::L1JetParticleCollection> tok_L1extFwdJet_;

  edm::EDGetTokenT<l1extra::L1MuonParticleCollection> tok_L1extMu_;
  edm::EDGetTokenT<l1extra::L1EmParticleCollection> tok_L1extIsoEm_;
  edm::EDGetTokenT<l1extra::L1EmParticleCollection> tok_L1extNoIsoEm_;

  edm::EDGetTokenT<reco::CaloJetCollection> tok_jets_;
  edm::EDGetTokenT<HBHERecHitCollection> tok_hbhe_;

  edm::EDGetTokenT<reco::TrackCollection> tok_genTrack_;
  edm::EDGetTokenT<reco::VertexCollection> tok_recVtx_;
  edm::EDGetTokenT<reco::BeamSpot> tok_bs_;

  edm::EDGetTokenT<EcalRecHitCollection> tok_EB_;
  edm::EDGetTokenT<EcalRecHitCollection> tok_EE_;
  edm::EDGetTokenT<edm::SimTrackContainer> tok_simTk_;
  edm::EDGetTokenT<edm::SimVertexContainer> tok_simVtx_;
  edm::EDGetTokenT<edm::PCaloHitContainer> tok_caloEB_;
  edm::EDGetTokenT<edm::PCaloHitContainer> tok_caloEE_;
  edm::EDGetTokenT<edm::PCaloHitContainer> tok_caloHH_;

  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom_;
  edm::ESGetToken<CaloTopology, CaloTopologyRecord> tok_caloTopology_;
  edm::ESGetToken<HcalTopology, HcalRecNumberingRecord> tok_topo_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> tok_magField_;
  edm::ESGetToken<EcalChannelStatus, EcalChannelStatusRcd> tok_ecalChStatus_;
  edm::ESGetToken<EcalSeverityLevelAlgo, EcalSeverityLevelAlgoRcd> tok_sevlv_;
  edm::ESGetToken<EcalTrigTowerConstituentsMap, IdealGeometryRecord> tok_htmap_;
  static constexpr size_t NPBins = 15;
  static constexpr size_t NEtaBins = 3;
  double genPartPBins[NPBins + 1], genPartEtaBins[NEtaBins + 1];

  TH1F *h_maxNearP15x15[NPBins][NEtaBins], *h_maxNearP21x21[NPBins][NEtaBins], *h_maxNearP25x25[NPBins][NEtaBins],
      *h_maxNearP31x31[NPBins][NEtaBins];
  TH1I *h_L1AlgoNames;
  TH1F *h_PVTracksWt;
  TH1F *h_nTracks;
  TH1F *h_recPt_0, *h_recP_0, *h_recEta_0, *h_recPhi_0;
  TH2F *h_recEtaPt_0, *h_recEtaP_0;
  TH1F *h_recPt_1, *h_recP_1, *h_recEta_1, *h_recPhi_1;
  TH2F *h_recEtaPt_1, *h_recEtaP_1;
  TH1F *h_recPt_2, *h_recP_2, *h_recEta_2, *h_recPhi_2;
  TH2F *h_recEtaPt_2, *h_recEtaP_2;

  TTree *tree_;

  int t_nTracks;
  int t_RunNo, t_EvtNo, t_Lumi, t_Bunch;
  std::vector<std::string> t_L1AlgoNames;
  std::vector<int> t_L1PreScale;
  int t_L1Decision[128];

  std::vector<double> t_L1CenJetPt, t_L1CenJetEta, t_L1CenJetPhi;
  std::vector<double> t_L1FwdJetPt, t_L1FwdJetEta, t_L1FwdJetPhi;
  std::vector<double> t_L1TauJetPt, t_L1TauJetEta, t_L1TauJetPhi;
  std::vector<double> t_L1MuonPt, t_L1MuonEta, t_L1MuonPhi;
  std::vector<double> t_L1IsoEMPt, t_L1IsoEMEta, t_L1IsoEMPhi;
  std::vector<double> t_L1NonIsoEMPt, t_L1NonIsoEMEta, t_L1NonIsoEMPhi;
  std::vector<double> t_L1METPt, t_L1METEta, t_L1METPhi;

  std::vector<double> t_PVx, t_PVy, t_PVz, t_PVTracksSumPt;
  std::vector<double> t_PVTracksSumPtWt, t_PVTracksSumPtHP, t_PVTracksSumPtHPWt;
  std::vector<int> t_PVisValid, t_PVNTracks, t_PVNTracksWt, t_PVndof;
  std::vector<int> t_PVNTracksHP, t_PVNTracksHPWt;

  std::vector<double> t_jetPt, t_jetEta, t_jetPhi;
  std::vector<double> t_nTrksJetCalo, t_nTrksJetVtx;

  std::vector<double> t_trackPAll, t_trackEtaAll, t_trackPhiAll, t_trackPdgIdAll;
  std::vector<double> t_trackPtAll;
  std::vector<double> t_trackDxyAll, t_trackDzAll, t_trackDxyPVAll, t_trackDzPVAll, t_trackChiSqAll;

  std::vector<double> t_trackP, t_trackPt, t_trackEta, t_trackPhi;
  std::vector<double> t_trackEcalEta, t_trackEcalPhi, t_trackHcalEta, t_trackHcalPhi;
  std::vector<double> t_trackDxy, t_trackDxyBS, t_trackDz, t_trackDzBS;
  std::vector<double> t_trackDxyPV, t_trackDzPV;
  std::vector<double> t_trackChiSq;
  std::vector<int> t_trackPVIdx;

  std::vector<int> t_NLayersCrossed, t_trackNOuterHits;
  std::vector<int> t_trackHitsTOB, t_trackHitsTEC;
  std::vector<int> t_trackHitInMissTOB, t_trackHitInMissTEC, t_trackHitInMissTIB, t_trackHitInMissTID,
      t_trackHitInMissTIBTID;
  std::vector<int> t_trackHitOutMissTOB, t_trackHitOutMissTEC, t_trackHitOutMissTIB, t_trackHitOutMissTID,
      t_trackHitOutMissTOBTEC;
  std::vector<int> t_trackHitInMeasTOB, t_trackHitInMeasTEC, t_trackHitInMeasTIB, t_trackHitInMeasTID;
  std::vector<int> t_trackHitOutMeasTOB, t_trackHitOutMeasTEC, t_trackHitOutMeasTIB, t_trackHitOutMeasTID;
  std::vector<double> t_trackOutPosOutHitDr, t_trackL;

  std::vector<double> t_maxNearP31x31;
  std::vector<double> t_maxNearP21x21;
  std::vector<int> t_ecalSpike11x11;

  std::vector<double> t_e7x7, t_e9x9, t_e11x11, t_e15x15;
  std::vector<double> t_e7x7_10Sig, t_e9x9_10Sig, t_e11x11_10Sig, t_e15x15_10Sig;
  std::vector<double> t_e7x7_15Sig, t_e9x9_15Sig, t_e11x11_15Sig, t_e15x15_15Sig;
  std::vector<double> t_e7x7_20Sig, t_e9x9_20Sig, t_e11x11_20Sig, t_e15x15_20Sig;
  std::vector<double> t_e7x7_25Sig, t_e9x9_25Sig, t_e11x11_25Sig, t_e15x15_25Sig;
  std::vector<double> t_e7x7_30Sig, t_e9x9_30Sig, t_e11x11_30Sig, t_e15x15_30Sig;

  std::vector<double> t_esimPdgId, t_simTrackP, t_trkEcalEne;
  std::vector<double> t_esim7x7, t_esim9x9, t_esim11x11, t_esim15x15;
  std::vector<double> t_esim7x7Matched, t_esim9x9Matched, t_esim11x11Matched, t_esim15x15Matched;
  std::vector<double> t_esim7x7Rest, t_esim9x9Rest, t_esim11x11Rest, t_esim15x15Rest;
  std::vector<double> t_esim7x7Photon, t_esim9x9Photon, t_esim11x11Photon, t_esim15x15Photon;
  std::vector<double> t_esim7x7NeutHad, t_esim9x9NeutHad, t_esim11x11NeutHad, t_esim15x15NeutHad;
  std::vector<double> t_esim7x7CharHad, t_esim9x9CharHad, t_esim11x11CharHad, t_esim15x15CharHad;

  std::vector<double> t_maxNearHcalP3x3, t_maxNearHcalP5x5, t_maxNearHcalP7x7;
  std::vector<double> t_h3x3, t_h5x5, t_h7x7;
  std::vector<double> t_h3x3Sig, t_h5x5Sig, t_h7x7Sig;
  std::vector<int> t_infoHcal;

  std::vector<double> t_trkHcalEne;
  std::vector<double> t_hsim3x3, t_hsim5x5, t_hsim7x7;
  std::vector<double> t_hsim3x3Matched, t_hsim5x5Matched, t_hsim7x7Matched;
  std::vector<double> t_hsim3x3Rest, t_hsim5x5Rest, t_hsim7x7Rest;
  std::vector<double> t_hsim3x3Photon, t_hsim5x5Photon, t_hsim7x7Photon;
  std::vector<double> t_hsim3x3NeutHad, t_hsim5x5NeutHad, t_hsim7x7NeutHad;
  std::vector<double> t_hsim3x3CharHad, t_hsim5x5CharHad, t_hsim7x7CharHad;
};

static const bool useL1EventSetup(true);
static const bool useL1GtTriggerMenuLite(true);

IsolatedTracksNxN::IsolatedTracksNxN(const edm::ParameterSet &iConfig)
    : trackerHitAssociatorConfig_(consumesCollector()),
      doMC_(iConfig.getUntrackedParameter<bool>("doMC", false)),
      writeAllTracks_(iConfig.getUntrackedParameter<bool>("writeAllTracks", false)),
      myverbose_(iConfig.getUntrackedParameter<int>("verbosity", 5)),
      pvTracksPtMin_(iConfig.getUntrackedParameter<double>("pvTracksPtMin", 1.0)),
      debugTrks_(iConfig.getUntrackedParameter<int>("debugTracks", 0)),
      printTrkHitPattern_(iConfig.getUntrackedParameter<bool>("printTrkHitPattern", false)),
      minTrackP_(iConfig.getUntrackedParameter<double>("minTrackP", 1.0)),
      maxTrackEta_(iConfig.getUntrackedParameter<double>("maxTrackEta", 5.0)),
      debugL1Info_(iConfig.getUntrackedParameter<bool>("debugL1Info", false)),
      L1TriggerAlgoInfo_(iConfig.getUntrackedParameter<bool>("l1TriggerAlgoInfo", false)),
      tMinE_(iConfig.getUntrackedParameter<double>("timeMinCutECAL", -500.)),
      tMaxE_(iConfig.getUntrackedParameter<double>("timeMaxCutECAL", 500.)),
      tMinH_(iConfig.getUntrackedParameter<double>("timeMinCutHCAL", -500.)),
      tMaxH_(iConfig.getUntrackedParameter<double>("timeMaxCutHCAL", 500.)) {
  if (L1TriggerAlgoInfo_) {
    m_l1GtUtils = std::make_unique<L1GtUtils>(
        iConfig, consumesCollector(), useL1GtTriggerMenuLite, *this, L1GtUtils::UseEventSetupIn::Event);
  }

  usesResource(TFileService::kSharedResource);

  //now do what ever initialization is needed

  edm::InputTag L1extraTauJetSource_ = iConfig.getParameter<edm::InputTag>("l1extraTauJetSource");
  edm::InputTag L1extraCenJetSource_ = iConfig.getParameter<edm::InputTag>("l1extraCenJetSource");
  edm::InputTag L1extraFwdJetSource_ = iConfig.getParameter<edm::InputTag>("l1extraFwdJetSource");
  edm::InputTag L1extraMuonSource_ = iConfig.getParameter<edm::InputTag>("l1extraMuonSource");
  edm::InputTag L1extraIsoEmSource_ = iConfig.getParameter<edm::InputTag>("l1extraIsoEmSource");
  edm::InputTag L1extraNonIsoEmSource_ = iConfig.getParameter<edm::InputTag>("l1extraNonIsoEmSource");
  edm::InputTag L1GTReadoutRcdSource_ = iConfig.getParameter<edm::InputTag>("l1GTReadoutRcdSource");
  edm::InputTag L1GTObjectMapRcdSource_ = iConfig.getParameter<edm::InputTag>("l1GTObjectMapRcdSource");
  edm::InputTag JetSrc_ = iConfig.getParameter<edm::InputTag>("jetSource");
  edm::InputTag JetExtender_ = iConfig.getParameter<edm::InputTag>("jetExtender");
  edm::InputTag HBHERecHitSource_ = iConfig.getParameter<edm::InputTag>("hbheRecHitSource");

  // define tokens for access
  tok_L1extTauJet_ = consumes<l1extra::L1JetParticleCollection>(L1extraTauJetSource_);
  tok_L1extCenJet_ = consumes<l1extra::L1JetParticleCollection>(L1extraCenJetSource_);
  tok_L1extFwdJet_ = consumes<l1extra::L1JetParticleCollection>(L1extraFwdJetSource_);
  tok_L1extMu_ = consumes<l1extra::L1MuonParticleCollection>(L1extraMuonSource_);
  tok_L1extIsoEm_ = consumes<l1extra::L1EmParticleCollection>(L1extraIsoEmSource_);
  tok_L1extNoIsoEm_ = consumes<l1extra::L1EmParticleCollection>(L1extraNonIsoEmSource_);
  tok_jets_ = consumes<reco::CaloJetCollection>(JetSrc_);
  tok_hbhe_ = consumes<HBHERecHitCollection>(HBHERecHitSource_);

  tok_genTrack_ = consumes<reco::TrackCollection>(edm::InputTag("generalTracks"));
  tok_recVtx_ = consumes<reco::VertexCollection>(edm::InputTag("offlinePrimaryVertices"));
  tok_bs_ = consumes<reco::BeamSpot>(edm::InputTag("offlineBeamSpot"));
  tok_EB_ = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit", "EcalRecHitsEB"));
  tok_EE_ = consumes<EcalRecHitCollection>(edm::InputTag("ecalRecHit", "EcalRecHitsEE"));

  tok_simTk_ = consumes<edm::SimTrackContainer>(edm::InputTag("g4SimHits"));
  tok_simVtx_ = consumes<edm::SimVertexContainer>(edm::InputTag("g4SimHits"));
  tok_caloEB_ = consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits", "EcalHitsEB"));
  tok_caloEE_ = consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits", "EcalHitsEE"));
  tok_caloHH_ = consumes<edm::PCaloHitContainer>(edm::InputTag("g4SimHits", "HcalHits"));

  if (myverbose_ >= 0) {
    edm::LogVerbatim("IsoTrack") << "Parameters read from config file \n"
                                 << " doMC        " << doMC_ << "\t myverbose " << myverbose_ << "\t minTrackP "
                                 << minTrackP_ << "\t maxTrackEta " << maxTrackEta_ << "\t tMinE " << tMinE_
                                 << "\t tMaxE " << tMaxE_ << "\t tMinH " << tMinH_ << "\t tMaxH " << tMaxH_
                                 << "\n debugL1Info " << debugL1Info_ << "\t L1TriggerAlgoInfo " << L1TriggerAlgoInfo_
                                 << "\n";
  }

  tok_geom_ = esConsumes<CaloGeometry, CaloGeometryRecord>();
  tok_caloTopology_ = esConsumes<CaloTopology, CaloTopologyRecord>();
  tok_topo_ = esConsumes<HcalTopology, HcalRecNumberingRecord>();
  tok_magField_ = esConsumes<MagneticField, IdealMagneticFieldRecord>();
  tok_ecalChStatus_ = esConsumes<EcalChannelStatus, EcalChannelStatusRcd>();
  tok_sevlv_ = esConsumes<EcalSeverityLevelAlgo, EcalSeverityLevelAlgoRcd>();
  tok_htmap_ = esConsumes<EcalTrigTowerConstituentsMap, IdealGeometryRecord>();
}

IsolatedTracksNxN::~IsolatedTracksNxN() {}

void IsolatedTracksNxN::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<bool>("doMC", false);
  desc.addUntracked<bool>("writeAllTracks", false);
  desc.addUntracked<int>("verbosity", 1);
  desc.addUntracked<double>("pvTracksPtMin", 0.200);
  desc.addUntracked<int>("debugTracks", 0);
  desc.addUntracked<bool>("printTrkHitPattern", true);
  desc.addUntracked<double>("minTrackP", 1.0);
  desc.addUntracked<double>("maxTrackEta", 2.6);
  desc.addUntracked<bool>("debugL1Info", false);
  desc.addUntracked<bool>("l1TriggerAlgoInfo", false);
  desc.add<edm::InputTag>("l1extraTauJetSource", edm::InputTag("l1extraParticles", "Tau"));
  desc.add<edm::InputTag>("l1extraCenJetSource", edm::InputTag("l1extraParticles", "Central"));
  desc.add<edm::InputTag>("l1extraFwdJetSource", edm::InputTag("l1extraParticles", "Forward"));
  desc.add<edm::InputTag>("l1extraMuonSource", edm::InputTag("l1extraParticles"));
  desc.add<edm::InputTag>("l1extraIsoEmSource", edm::InputTag("l1extraParticles", "Isolated"));
  desc.add<edm::InputTag>("l1extraNonIsoEmSource", edm::InputTag("l1extraParticles", "NonIsolated"));
  desc.add<edm::InputTag>("l1GTReadoutRcdSource", edm::InputTag("gtDigis"));
  desc.add<edm::InputTag>("l1GTObjectMapRcdSource", edm::InputTag("hltL1GtObjectMap"));
  desc.add<edm::InputTag>("jetSource", edm::InputTag("iterativeCone5CaloJets"));
  desc.add<edm::InputTag>("jetExtender", edm::InputTag("iterativeCone5JetExtender"));
  desc.add<edm::InputTag>("hbheRecHitSource", edm::InputTag("hbhereco"));
  desc.addUntracked<double>("maxNearTrackPT", 1.0);
  desc.addUntracked<double>("timeMinCutECAL", -500.0);
  desc.addUntracked<double>("timeMaxCutECAL", 500.0);
  desc.addUntracked<double>("timeMinCutHCAL", -500.0);
  desc.addUntracked<double>("timeMaxCutHCAL", 500.0);
  descriptions.add("isolatedTracksNxN", desc);
}

void IsolatedTracksNxN::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  bool haveIsoTrack = false;

  const MagneticField *bField = &iSetup.getData(tok_magField_);

  clearTreeVectors();

  t_RunNo = iEvent.id().run();
  t_EvtNo = iEvent.id().event();
  t_Lumi = iEvent.luminosityBlock();
  t_Bunch = iEvent.bunchCrossing();

  ++nEventProc_;

  edm::Handle<reco::TrackCollection> trkCollection;
  iEvent.getByToken(tok_genTrack_, trkCollection);
  if (debugTrks_ > 1) {
    edm::LogVerbatim("IsoTrack") << "Track Collection: ";
    edm::LogVerbatim("IsoTrack") << "Number of Tracks " << trkCollection->size();
  }
  std::string theTrackQuality = "highPurity";
  reco::TrackBase::TrackQuality trackQuality_ = reco::TrackBase::qualityByName(theTrackQuality);

  //===================== save L1 Trigger information =======================
  if (L1TriggerAlgoInfo_) {
    m_l1GtUtils->getL1GtRunCache(iEvent, iSetup, useL1EventSetup, useL1GtTriggerMenuLite);

    int iErrorCode = -1;
    int l1ConfCode = -1;
    const bool l1Conf = m_l1GtUtils->availableL1Configuration(iErrorCode, l1ConfCode);
    if (!l1Conf) {
      edm::LogVerbatim("IsoTrack")
          << "\nL1 configuration code:" << l1ConfCode << "\nNo valid L1 trigger configuration available."
          << "\nSee text above for error code interpretation"
          << "\nNo return here, in order to test each method, protected against configuration error.";
    }

    const L1GtTriggerMenu *m_l1GtMenu = m_l1GtUtils->ptrL1TriggerMenuEventSetup(iErrorCode);
    const AlgorithmMap &algorithmMap = m_l1GtMenu->gtAlgorithmMap();
    const std::string &menuName = m_l1GtMenu->gtTriggerMenuName();

    if (!initL1_) {
      initL1_ = true;
      edm::LogVerbatim("IsoTrack") << "menuName " << menuName;
      for (CItAlgo itAlgo = algorithmMap.begin(); itAlgo != algorithmMap.end(); itAlgo++) {
        std::string algName = itAlgo->first;
        int algBitNumber = (itAlgo->second).algoBitNumber();
        l1AlgoMap_.insert(std::pair<std::pair<unsigned int, std::string>, int>(
            std::pair<unsigned int, std::string>(algBitNumber, algName), 0));
      }
      std::map<std::pair<unsigned int, std::string>, int>::iterator itr;
      for (itr = l1AlgoMap_.begin(); itr != l1AlgoMap_.end(); itr++) {
        edm::LogVerbatim("IsoTrack") << " ********** " << (itr->first).first << " " << (itr->first).second << " "
                                     << itr->second;
      }
    }

    std::vector<int> algbits;
    for (CItAlgo itAlgo = algorithmMap.begin(); itAlgo != algorithmMap.end(); itAlgo++) {
      std::string algName = itAlgo->first;
      int algBitNumber = (itAlgo->second).algoBitNumber();
      bool decision = m_l1GtUtils->decision(iEvent, itAlgo->first, iErrorCode);
      int preScale = m_l1GtUtils->prescaleFactor(iEvent, itAlgo->first, iErrorCode);

      // save the algo names which fired
      if (decision) {
        l1AlgoMap_[std::pair<unsigned int, std::string>(algBitNumber, algName)] += 1;
        h_L1AlgoNames->Fill(algBitNumber);
        t_L1AlgoNames.push_back(itAlgo->first);
        t_L1PreScale.push_back(preScale);
        t_L1Decision[algBitNumber] = 1;
        algbits.push_back(algBitNumber);
      }
    }

    if (debugL1Info_) {
      for (unsigned int ii = 0; ii < t_L1AlgoNames.size(); ii++) {
        edm::LogVerbatim("IsoTrack") << ii << " " << t_L1AlgoNames[ii] << " " << t_L1PreScale[ii] << " " << algbits[ii];
      }
      for (int i = 0; i < 128; ++i) {
        edm::LogVerbatim("IsoTrack") << "L1Decision: " << i << ":" << t_L1Decision[i];
      }
    }

    // L1Taus
    const edm::Handle<l1extra::L1JetParticleCollection> &l1TauHandle = iEvent.getHandle(tok_L1extTauJet_);
    l1extra::L1JetParticleCollection::const_iterator itr;
    int iL1Obj = 0;
    for (itr = l1TauHandle->begin(), iL1Obj = 0; itr != l1TauHandle->end(); ++itr, iL1Obj++) {
      if (iL1Obj < 1) {
        t_L1TauJetPt.push_back(itr->pt());
        t_L1TauJetEta.push_back(itr->eta());
        t_L1TauJetPhi.push_back(itr->phi());
      }
      if (debugL1Info_) {
        edm::LogVerbatim("IsoTrack") << "tauJ p/pt  " << itr->momentum() << " " << itr->pt() << "  eta/phi "
                                     << itr->eta() << " " << itr->phi();
      }
    }

    // L1 Central Jets
    const edm::Handle<l1extra::L1JetParticleCollection> &l1CenJetHandle = iEvent.getHandle(tok_L1extCenJet_);
    for (itr = l1CenJetHandle->begin(), iL1Obj = 0; itr != l1CenJetHandle->end(); ++itr, iL1Obj++) {
      if (iL1Obj < 1) {
        t_L1CenJetPt.push_back(itr->pt());
        t_L1CenJetEta.push_back(itr->eta());
        t_L1CenJetPhi.push_back(itr->phi());
      }
      if (debugL1Info_) {
        edm::LogVerbatim("IsoTrack") << "cenJ p/pt     " << itr->momentum() << " " << itr->pt() << "  eta/phi "
                                     << itr->eta() << " " << itr->phi();
      }
    }

    // L1 Forward Jets
    const edm::Handle<l1extra::L1JetParticleCollection> &l1FwdJetHandle = iEvent.getHandle(tok_L1extFwdJet_);
    for (itr = l1FwdJetHandle->begin(), iL1Obj = 0; itr != l1FwdJetHandle->end(); ++itr, iL1Obj++) {
      if (iL1Obj < 1) {
        t_L1FwdJetPt.push_back(itr->pt());
        t_L1FwdJetEta.push_back(itr->eta());
        t_L1FwdJetPhi.push_back(itr->phi());
      }
      if (debugL1Info_) {
        edm::LogVerbatim("IsoTrack") << "fwdJ p/pt     " << itr->momentum() << " " << itr->pt() << "  eta/phi "
                                     << itr->eta() << " " << itr->phi();
      }
    }

    // L1 Isolated EM onjects
    l1extra::L1EmParticleCollection::const_iterator itrEm;
    const edm::Handle<l1extra::L1EmParticleCollection> &l1IsoEmHandle = iEvent.getHandle(tok_L1extIsoEm_);
    for (itrEm = l1IsoEmHandle->begin(), iL1Obj = 0; itrEm != l1IsoEmHandle->end(); ++itrEm, iL1Obj++) {
      if (iL1Obj < 1) {
        t_L1IsoEMPt.push_back(itrEm->pt());
        t_L1IsoEMEta.push_back(itrEm->eta());
        t_L1IsoEMPhi.push_back(itrEm->phi());
      }
      if (debugL1Info_) {
        edm::LogVerbatim("IsoTrack") << "isoEm p/pt    " << itrEm->momentum() << " " << itrEm->pt() << "  eta/phi "
                                     << itrEm->eta() << " " << itrEm->phi();
      }
    }

    // L1 Non-Isolated EM onjects
    const edm::Handle<l1extra::L1EmParticleCollection> &l1NonIsoEmHandle = iEvent.getHandle(tok_L1extNoIsoEm_);
    for (itrEm = l1NonIsoEmHandle->begin(), iL1Obj = 0; itrEm != l1NonIsoEmHandle->end(); ++itrEm, iL1Obj++) {
      if (iL1Obj < 1) {
        t_L1NonIsoEMPt.push_back(itrEm->pt());
        t_L1NonIsoEMEta.push_back(itrEm->eta());
        t_L1NonIsoEMPhi.push_back(itrEm->phi());
      }
      if (debugL1Info_) {
        edm::LogVerbatim("IsoTrack") << "nonIsoEm p/pt " << itrEm->momentum() << " " << itrEm->pt() << "  eta/phi "
                                     << itrEm->eta() << " " << itrEm->phi();
      }
    }

    // L1 Muons
    l1extra::L1MuonParticleCollection::const_iterator itrMu;
    const edm::Handle<l1extra::L1MuonParticleCollection> &l1MuHandle = iEvent.getHandle(tok_L1extMu_);
    for (itrMu = l1MuHandle->begin(), iL1Obj = 0; itrMu != l1MuHandle->end(); ++itrMu, iL1Obj++) {
      if (iL1Obj < 1) {
        t_L1MuonPt.push_back(itrMu->pt());
        t_L1MuonEta.push_back(itrMu->eta());
        t_L1MuonPhi.push_back(itrMu->phi());
      }
      if (debugL1Info_) {
        edm::LogVerbatim("IsoTrack") << "l1muon p/pt   " << itrMu->momentum() << " " << itrMu->pt() << "  eta/phi "
                                     << itrMu->eta() << " " << itrMu->phi();
      }
    }
  }

  //============== store the information about all the Non-Fake vertices ===============

  const edm::Handle<reco::VertexCollection> &recVtxs = iEvent.getHandle(tok_recVtx_);

  std::vector<reco::Track> svTracks;
  math::XYZPoint leadPV(0, 0, 0);
  double sumPtMax = -1.0;
  for (unsigned int ind = 0; ind < recVtxs->size(); ind++) {
    if (!((*recVtxs)[ind].isFake())) {
      double vtxTrkSumPt = 0.0, vtxTrkSumPtWt = 0.0;
      int vtxTrkNWt = 0;
      double vtxTrkSumPtHP = 0.0, vtxTrkSumPtHPWt = 0.0;
      int vtxTrkNHP = 0, vtxTrkNHPWt = 0;

      reco::Vertex::trackRef_iterator vtxTrack = (*recVtxs)[ind].tracks_begin();

      for (vtxTrack = (*recVtxs)[ind].tracks_begin(); vtxTrack != (*recVtxs)[ind].tracks_end(); vtxTrack++) {
        if ((*vtxTrack)->pt() < pvTracksPtMin_)
          continue;

        bool trkQuality = (*vtxTrack)->quality(trackQuality_);

        vtxTrkSumPt += (*vtxTrack)->pt();

        vtxTrkSumPt += (*vtxTrack)->pt();
        if (trkQuality) {
          vtxTrkSumPtHP += (*vtxTrack)->pt();
          vtxTrkNHP++;
        }

        double weight = (*recVtxs)[ind].trackWeight(*vtxTrack);
        h_PVTracksWt->Fill(weight);
        if (weight > 0.5) {
          vtxTrkSumPtWt += (*vtxTrack)->pt();
          vtxTrkNWt++;
          if (trkQuality) {
            vtxTrkSumPtHPWt += (*vtxTrack)->pt();
            vtxTrkNHPWt++;
          }
        }
      }

      if (vtxTrkSumPt > sumPtMax) {
        sumPtMax = vtxTrkSumPt;
        leadPV = math::XYZPoint((*recVtxs)[ind].x(), (*recVtxs)[ind].y(), (*recVtxs)[ind].z());
      }

      t_PVx.push_back((*recVtxs)[ind].x());
      t_PVy.push_back((*recVtxs)[ind].y());
      t_PVz.push_back((*recVtxs)[ind].z());
      t_PVisValid.push_back((*recVtxs)[ind].isValid());
      t_PVNTracks.push_back((*recVtxs)[ind].tracksSize());
      t_PVndof.push_back((*recVtxs)[ind].ndof());
      t_PVNTracksWt.push_back(vtxTrkNWt);
      t_PVTracksSumPt.push_back(vtxTrkSumPt);
      t_PVTracksSumPtWt.push_back(vtxTrkSumPtWt);

      t_PVNTracksHP.push_back(vtxTrkNHP);
      t_PVNTracksHPWt.push_back(vtxTrkNHPWt);
      t_PVTracksSumPtHP.push_back(vtxTrkSumPtHP);
      t_PVTracksSumPtHPWt.push_back(vtxTrkSumPtHPWt);

      if (myverbose_ == 4) {
        edm::LogVerbatim("IsoTrack") << "PV " << ind << " isValid " << (*recVtxs)[ind].isValid() << " isFake "
                                     << (*recVtxs)[ind].isFake() << " hasRefittedTracks() " << ind << " "
                                     << (*recVtxs)[ind].hasRefittedTracks() << " refittedTrksSize "
                                     << (*recVtxs)[ind].refittedTracks().size() << "  tracksSize() "
                                     << (*recVtxs)[ind].tracksSize() << " sumPt " << vtxTrkSumPt;
      }
    }  // if vtx is not Fake
  }    // loop over PVs
  //===================================================================================

  // Get the beamspot
  const edm::Handle<reco::BeamSpot> &beamSpotH = iEvent.getHandle(tok_bs_);
  math::XYZPoint bspot;
  bspot = (beamSpotH.isValid()) ? beamSpotH->position() : math::XYZPoint(0, 0, 0);

  //=====================================================================

  const edm::Handle<reco::CaloJetCollection> &jets = iEvent.getHandle(tok_jets_);

  for (unsigned int ijet = 0; ijet < (*jets).size(); ijet++) {
    t_jetPt.push_back((*jets)[ijet].pt());
    t_jetEta.push_back((*jets)[ijet].eta());
    t_jetPhi.push_back((*jets)[ijet].phi());
    t_nTrksJetVtx.push_back(-1.0);
    t_nTrksJetCalo.push_back(-1.0);
  }

  //===================================================================================

  // get handles to calogeometry and calotopology
  const CaloGeometry *geo = &iSetup.getData(tok_geom_);
  const CaloTopology *caloTopology = &iSetup.getData(tok_caloTopology_);
  const HcalTopology *theHBHETopology = &iSetup.getData(tok_topo_);

  edm::Handle<EcalRecHitCollection> barrelRecHitsHandle;
  iEvent.getByToken(tok_EB_, barrelRecHitsHandle);
  edm::Handle<EcalRecHitCollection> endcapRecHitsHandle;
  iEvent.getByToken(tok_EE_, endcapRecHitsHandle);

  // Retrieve the good/bad ECAL channels from the DB
  const EcalChannelStatus *theEcalChStatus = &iSetup.getData(tok_ecalChStatus_);
  const EcalSeverityLevelAlgo *sevlv = &iSetup.getData(tok_sevlv_);

  // Retrieve trigger tower map
  const EcalTrigTowerConstituentsMap &ttMap = iSetup.getData(tok_htmap_);

  edm::Handle<HBHERecHitCollection> hbhe;
  iEvent.getByToken(tok_hbhe_, hbhe);
  if (!hbhe.isValid()) {
    ++nbad_;
    if (nbad_ < 10)
      edm::LogVerbatim("IsoTrack") << "No HBHE rechit collection";
    return;
  }
  const HBHERecHitCollection Hithbhe = *(hbhe.product());

  //get Handles to SimTracks and SimHits
  edm::Handle<edm::SimTrackContainer> SimTk;
  if (doMC_)
    iEvent.getByToken(tok_simTk_, SimTk);

  edm::Handle<edm::SimVertexContainer> SimVtx;
  if (doMC_)
    iEvent.getByToken(tok_simVtx_, SimVtx);

  //get Handles to PCaloHitContainers of eb/ee/hbhe
  edm::Handle<edm::PCaloHitContainer> pcaloeb;
  if (doMC_)
    iEvent.getByToken(tok_caloEB_, pcaloeb);

  edm::Handle<edm::PCaloHitContainer> pcaloee;
  if (doMC_)
    iEvent.getByToken(tok_caloEE_, pcaloee);

  edm::Handle<edm::PCaloHitContainer> pcalohh;
  if (doMC_)
    iEvent.getByToken(tok_caloHH_, pcalohh);

  //associates tracker rechits/simhits to a track
  std::unique_ptr<TrackerHitAssociator> associate;
  if (doMC_)
    associate = std::make_unique<TrackerHitAssociator>(iEvent, trackerHitAssociatorConfig_);

  //===================================================================================

  h_nTracks->Fill(trkCollection->size());

  int nTracks = 0;

  t_nTracks = trkCollection->size();

  // get the list of DetIds closest to the impact point of track on surface calorimeters
  std::vector<spr::propagatedTrackID> trkCaloDets;
  spr::propagateCALO(trkCollection, geo, bField, theTrackQuality, trkCaloDets, false);
  std::vector<spr::propagatedTrackID>::const_iterator trkDetItr;

  if (myverbose_ > 2) {
    for (trkDetItr = trkCaloDets.begin(); trkDetItr != trkCaloDets.end(); trkDetItr++) {
      edm::LogVerbatim("IsoTrack") << trkDetItr->trkItr->p() << " " << trkDetItr->trkItr->eta() << " "
                                   << trkDetItr->okECAL << " " << trkDetItr->okHCAL;
      if (trkDetItr->okECAL) {
        if (trkDetItr->detIdECAL.subdetId() == EcalBarrel)
          edm::LogVerbatim("IsoTrack") << (EBDetId)trkDetItr->detIdECAL;
        else
          edm::LogVerbatim("IsoTrack") << (EEDetId)trkDetItr->detIdECAL;
      }
      if (trkDetItr->okHCAL)
        edm::LogVerbatim("IsoTrack") << (HcalDetId)trkDetItr->detIdHCAL;
    }
  }

  int nvtxTracks = 0;
  for (trkDetItr = trkCaloDets.begin(), nTracks = 0; trkDetItr != trkCaloDets.end(); trkDetItr++, nTracks++) {
    const reco::Track *pTrack = &(*(trkDetItr->trkItr));

    // find vertex index the track is associated with
    int pVtxTkId = -1;
    for (unsigned int ind = 0; ind < recVtxs->size(); ind++) {
      if (!((*recVtxs)[ind].isFake())) {
        reco::Vertex::trackRef_iterator vtxTrack = (*recVtxs)[ind].tracks_begin();
        for (vtxTrack = (*recVtxs)[ind].tracks_begin(); vtxTrack != (*recVtxs)[ind].tracks_end(); vtxTrack++) {
          const edm::RefToBase<reco::Track> &pvtxTrack = (*vtxTrack);
          if (pTrack == pvtxTrack.get()) {
            pVtxTkId = ind;
            break;
            if (myverbose_ > 2) {
              if (pTrack->pt() > 1.0) {
                edm::LogVerbatim("IsoTrack") << "Debug the track association with vertex ";
                edm::LogVerbatim("IsoTrack") << pTrack << " " << pvtxTrack.get();
                edm::LogVerbatim("IsoTrack")
                    << " trkVtxIndex " << nvtxTracks << " vtx " << ind << " pt " << pTrack->pt() << " eta "
                    << pTrack->eta() << " " << pTrack->pt() - pvtxTrack->pt() << " "
                    << pTrack->eta() - pvtxTrack->eta();
                nvtxTracks++;
              }
            }
          }
        }
      }
    }

    const reco::HitPattern &hitp = pTrack->hitPattern();
    int nLayersCrossed = hitp.trackerLayersWithMeasurement();
    int nOuterHits = hitp.stripTOBLayersWithMeasurement() + hitp.stripTECLayersWithMeasurement();

    bool ifGood = pTrack->quality(trackQuality_);
    double pt1 = pTrack->pt();
    double p1 = pTrack->p();
    double eta1 = pTrack->momentum().eta();
    double phi1 = pTrack->momentum().phi();
    double etaEcal1 = trkDetItr->etaECAL;
    double phiEcal1 = trkDetItr->phiECAL;
    double etaHcal1 = trkDetItr->etaHCAL;
    double phiHcal1 = trkDetItr->phiHCAL;
    double dxy1 = pTrack->dxy();
    double dz1 = pTrack->dz();
    double chisq1 = pTrack->normalizedChi2();
    double dxybs1 = beamSpotH.isValid() ? pTrack->dxy(bspot) : pTrack->dxy();
    double dzbs1 = beamSpotH.isValid() ? pTrack->dz(bspot) : pTrack->dz();
    double dxypv1 = pTrack->dxy();
    double dzpv1 = pTrack->dz();
    if (pVtxTkId >= 0) {
      math::XYZPoint thisTkPV =
          math::XYZPoint((*recVtxs)[pVtxTkId].x(), (*recVtxs)[pVtxTkId].y(), (*recVtxs)[pVtxTkId].z());
      dxypv1 = pTrack->dxy(thisTkPV);
      dzpv1 = pTrack->dz(thisTkPV);
    }

    h_recEtaPt_0->Fill(eta1, pt1);
    h_recEtaP_0->Fill(eta1, p1);
    h_recPt_0->Fill(pt1);
    h_recP_0->Fill(p1);
    h_recEta_0->Fill(eta1);
    h_recPhi_0->Fill(phi1);

    if (ifGood && nLayersCrossed > 7) {
      h_recEtaPt_1->Fill(eta1, pt1);
      h_recEtaP_1->Fill(eta1, p1);
      h_recPt_1->Fill(pt1);
      h_recP_1->Fill(p1);
      h_recEta_1->Fill(eta1);
      h_recPhi_1->Fill(phi1);
    }

    if (!ifGood)
      continue;

    if (writeAllTracks_ && p1 > 2.0 && nLayersCrossed > 7) {
      t_trackPAll.push_back(p1);
      t_trackEtaAll.push_back(eta1);
      t_trackPhiAll.push_back(phi1);
      t_trackPtAll.push_back(pt1);
      t_trackDxyAll.push_back(dxy1);
      t_trackDzAll.push_back(dz1);
      t_trackDxyPVAll.push_back(dxypv1);
      t_trackDzPVAll.push_back(dzpv1);
      t_trackChiSqAll.push_back(chisq1);
    }
    if (doMC_) {
      edm::SimTrackContainer::const_iterator matchedSimTrkAll =
          spr::matchedSimTrack(iEvent, SimTk, SimVtx, pTrack, *associate, false);
      if (writeAllTracks_ && matchedSimTrkAll != SimTk->end())
        t_trackPdgIdAll.push_back(matchedSimTrkAll->type());
    }

    if (pt1 > minTrackP_ && std::abs(eta1) < maxTrackEta_ && trkDetItr->okECAL) {
      double maxNearP31x31 = 999.0, maxNearP25x25 = 999.0, maxNearP21x21 = 999.0, maxNearP15x15 = 999.0;
      maxNearP31x31 = spr::chargeIsolationEcal(nTracks, trkCaloDets, geo, caloTopology, 15, 15);
      maxNearP25x25 = spr::chargeIsolationEcal(nTracks, trkCaloDets, geo, caloTopology, 12, 12);
      maxNearP21x21 = spr::chargeIsolationEcal(nTracks, trkCaloDets, geo, caloTopology, 10, 10);
      maxNearP15x15 = spr::chargeIsolationEcal(nTracks, trkCaloDets, geo, caloTopology, 7, 7);

      int iTrkEtaBin = -1, iTrkMomBin = -1;
      for (unsigned int ieta = 0; ieta < NEtaBins; ieta++) {
        if (std::abs(eta1) > genPartEtaBins[ieta] && std::abs(eta1) < genPartEtaBins[ieta + 1])
          iTrkEtaBin = ieta;
      }
      for (unsigned int ipt = 0; ipt < NPBins; ipt++) {
        if (p1 > genPartPBins[ipt] && p1 < genPartPBins[ipt + 1])
          iTrkMomBin = ipt;
      }
      if (iTrkMomBin >= 0 && iTrkEtaBin >= 0) {
        h_maxNearP31x31[iTrkMomBin][iTrkEtaBin]->Fill(maxNearP31x31);
        h_maxNearP25x25[iTrkMomBin][iTrkEtaBin]->Fill(maxNearP25x25);
        h_maxNearP21x21[iTrkMomBin][iTrkEtaBin]->Fill(maxNearP21x21);
        h_maxNearP15x15[iTrkMomBin][iTrkEtaBin]->Fill(maxNearP15x15);
      }
      if (maxNearP31x31 < 0.0 && nLayersCrossed > 7 && nOuterHits > 4) {
        h_recEtaPt_2->Fill(eta1, pt1);
        h_recEtaP_2->Fill(eta1, p1);
        h_recPt_2->Fill(pt1);
        h_recP_2->Fill(p1);
        h_recEta_2->Fill(eta1);
        h_recPhi_2->Fill(phi1);
      }

      // if charge isolated in ECAL, store the further quantities
      if (maxNearP31x31 < 1.0) {
        haveIsoTrack = true;

        // get the matching simTrack
        double simTrackP = -1;
        if (doMC_) {
          edm::SimTrackContainer::const_iterator matchedSimTrk =
              spr::matchedSimTrack(iEvent, SimTk, SimVtx, pTrack, *associate, false);
          if (matchedSimTrk != SimTk->end())
            simTrackP = matchedSimTrk->momentum().P();
        }

        // get ECal Tranverse Profile
        std::pair<double, bool> e7x7P, e9x9P, e11x11P, e15x15P;
        std::pair<double, bool> e7x7_10SigP, e9x9_10SigP, e11x11_10SigP, e15x15_10SigP;
        std::pair<double, bool> e7x7_15SigP, e9x9_15SigP, e11x11_15SigP, e15x15_15SigP;
        std::pair<double, bool> e7x7_20SigP, e9x9_20SigP, e11x11_20SigP, e15x15_20SigP;
        std::pair<double, bool> e7x7_25SigP, e9x9_25SigP, e11x11_25SigP, e15x15_25SigP;
        std::pair<double, bool> e7x7_30SigP, e9x9_30SigP, e11x11_30SigP, e15x15_30SigP;

        spr::caloSimInfo simInfo3x3, simInfo5x5, simInfo7x7, simInfo9x9;
        spr::caloSimInfo simInfo11x11, simInfo13x13, simInfo15x15, simInfo21x21, simInfo25x25, simInfo31x31;
        double trkEcalEne = 0;

        const DetId isoCell = trkDetItr->detIdECAL;
        e7x7P = spr::eECALmatrix(isoCell,
                                 barrelRecHitsHandle,
                                 endcapRecHitsHandle,
                                 *theEcalChStatus,
                                 geo,
                                 caloTopology,
                                 sevlv,
                                 3,
                                 3,
                                 -100.0,
                                 -100.0,
                                 tMinE_,
                                 tMaxE_);
        e9x9P = spr::eECALmatrix(isoCell,
                                 barrelRecHitsHandle,
                                 endcapRecHitsHandle,
                                 *theEcalChStatus,
                                 geo,
                                 caloTopology,
                                 sevlv,
                                 4,
                                 4,
                                 -100.0,
                                 -100.0,
                                 tMinE_,
                                 tMaxE_);
        e11x11P = spr::eECALmatrix(isoCell,
                                   barrelRecHitsHandle,
                                   endcapRecHitsHandle,
                                   *theEcalChStatus,
                                   geo,
                                   caloTopology,
                                   sevlv,
                                   5,
                                   5,
                                   -100.0,
                                   -100.0,
                                   tMinE_,
                                   tMaxE_);
        e15x15P = spr::eECALmatrix(isoCell,
                                   barrelRecHitsHandle,
                                   endcapRecHitsHandle,
                                   *theEcalChStatus,
                                   geo,
                                   caloTopology,
                                   sevlv,
                                   7,
                                   7,
                                   -100.0,
                                   -100.0,
                                   tMinE_,
                                   tMaxE_);

        e7x7_10SigP = spr::eECALmatrix(isoCell,
                                       barrelRecHitsHandle,
                                       endcapRecHitsHandle,
                                       *theEcalChStatus,
                                       geo,
                                       caloTopology,
                                       sevlv,
                                       3,
                                       3,
                                       0.030,
                                       0.150,
                                       tMinE_,
                                       tMaxE_);
        e9x9_10SigP = spr::eECALmatrix(isoCell,
                                       barrelRecHitsHandle,
                                       endcapRecHitsHandle,
                                       *theEcalChStatus,
                                       geo,
                                       caloTopology,
                                       sevlv,
                                       4,
                                       4,
                                       0.030,
                                       0.150,
                                       tMinE_,
                                       tMaxE_);
        e11x11_10SigP = spr::eECALmatrix(isoCell,
                                         barrelRecHitsHandle,
                                         endcapRecHitsHandle,
                                         *theEcalChStatus,
                                         geo,
                                         caloTopology,
                                         sevlv,
                                         5,
                                         5,
                                         0.030,
                                         0.150,
                                         tMinE_,
                                         tMaxE_);
        e15x15_10SigP = spr::eECALmatrix(isoCell,
                                         barrelRecHitsHandle,
                                         endcapRecHitsHandle,
                                         *theEcalChStatus,
                                         geo,
                                         caloTopology,
                                         sevlv,
                                         7,
                                         7,
                                         0.030,
                                         0.150,
                                         tMinE_,
                                         tMaxE_);

        e7x7_15SigP = spr::eECALmatrix(isoCell,
                                       barrelRecHitsHandle,
                                       endcapRecHitsHandle,
                                       *theEcalChStatus,
                                       geo,
                                       caloTopology,
                                       sevlv,
                                       ttMap,
                                       3,
                                       3,
                                       0.20,
                                       0.45,
                                       tMinE_,
                                       tMaxE_);
        e9x9_15SigP = spr::eECALmatrix(isoCell,
                                       barrelRecHitsHandle,
                                       endcapRecHitsHandle,
                                       *theEcalChStatus,
                                       geo,
                                       caloTopology,
                                       sevlv,
                                       ttMap,
                                       4,
                                       4,
                                       0.20,
                                       0.45,
                                       tMinE_,
                                       tMaxE_);
        e11x11_15SigP = spr::eECALmatrix(isoCell,
                                         barrelRecHitsHandle,
                                         endcapRecHitsHandle,
                                         *theEcalChStatus,
                                         geo,
                                         caloTopology,
                                         sevlv,
                                         ttMap,
                                         5,
                                         5,
                                         0.20,
                                         0.45,
                                         tMinE_,
                                         tMaxE_);
        e15x15_15SigP = spr::eECALmatrix(isoCell,
                                         barrelRecHitsHandle,
                                         endcapRecHitsHandle,
                                         *theEcalChStatus,
                                         geo,
                                         caloTopology,
                                         sevlv,
                                         ttMap,
                                         7,
                                         7,
                                         0.20,
                                         0.45,
                                         tMinE_,
                                         tMaxE_,
                                         false);

        e7x7_20SigP = spr::eECALmatrix(isoCell,
                                       barrelRecHitsHandle,
                                       endcapRecHitsHandle,
                                       *theEcalChStatus,
                                       geo,
                                       caloTopology,
                                       sevlv,
                                       3,
                                       3,
                                       0.060,
                                       0.300,
                                       tMinE_,
                                       tMaxE_);
        e9x9_20SigP = spr::eECALmatrix(isoCell,
                                       barrelRecHitsHandle,
                                       endcapRecHitsHandle,
                                       *theEcalChStatus,
                                       geo,
                                       caloTopology,
                                       sevlv,
                                       4,
                                       4,
                                       0.060,
                                       0.300,
                                       tMinE_,
                                       tMaxE_);
        e11x11_20SigP = spr::eECALmatrix(isoCell,
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
        e15x15_20SigP = spr::eECALmatrix(isoCell,
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

        e7x7_25SigP = spr::eECALmatrix(isoCell,
                                       barrelRecHitsHandle,
                                       endcapRecHitsHandle,
                                       *theEcalChStatus,
                                       geo,
                                       caloTopology,
                                       sevlv,
                                       3,
                                       3,
                                       0.075,
                                       0.375,
                                       tMinE_,
                                       tMaxE_);
        e9x9_25SigP = spr::eECALmatrix(isoCell,
                                       barrelRecHitsHandle,
                                       endcapRecHitsHandle,
                                       *theEcalChStatus,
                                       geo,
                                       caloTopology,
                                       sevlv,
                                       4,
                                       4,
                                       0.075,
                                       0.375,
                                       tMinE_,
                                       tMaxE_);
        e11x11_25SigP = spr::eECALmatrix(isoCell,
                                         barrelRecHitsHandle,
                                         endcapRecHitsHandle,
                                         *theEcalChStatus,
                                         geo,
                                         caloTopology,
                                         sevlv,
                                         5,
                                         5,
                                         0.075,
                                         0.375,
                                         tMinE_,
                                         tMaxE_);
        e15x15_25SigP = spr::eECALmatrix(isoCell,
                                         barrelRecHitsHandle,
                                         endcapRecHitsHandle,
                                         *theEcalChStatus,
                                         geo,
                                         caloTopology,
                                         sevlv,
                                         7,
                                         7,
                                         0.075,
                                         0.375,
                                         tMinE_,
                                         tMaxE_);

        e7x7_30SigP = spr::eECALmatrix(isoCell,
                                       barrelRecHitsHandle,
                                       endcapRecHitsHandle,
                                       *theEcalChStatus,
                                       geo,
                                       caloTopology,
                                       sevlv,
                                       3,
                                       3,
                                       0.090,
                                       0.450,
                                       tMinE_,
                                       tMaxE_);
        e9x9_30SigP = spr::eECALmatrix(isoCell,
                                       barrelRecHitsHandle,
                                       endcapRecHitsHandle,
                                       *theEcalChStatus,
                                       geo,
                                       caloTopology,
                                       sevlv,
                                       4,
                                       4,
                                       0.090,
                                       0.450,
                                       tMinE_,
                                       tMaxE_);
        e11x11_30SigP = spr::eECALmatrix(isoCell,
                                         barrelRecHitsHandle,
                                         endcapRecHitsHandle,
                                         *theEcalChStatus,
                                         geo,
                                         caloTopology,
                                         sevlv,
                                         5,
                                         5,
                                         0.090,
                                         0.450,
                                         tMinE_,
                                         tMaxE_);
        e15x15_30SigP = spr::eECALmatrix(isoCell,
                                         barrelRecHitsHandle,
                                         endcapRecHitsHandle,
                                         *theEcalChStatus,
                                         geo,
                                         caloTopology,
                                         sevlv,
                                         7,
                                         7,
                                         0.090,
                                         0.450,
                                         tMinE_,
                                         tMaxE_);
        if (myverbose_ == 2) {
          edm::LogVerbatim("IsoTrack") << "clean  ecal rechit ";
          edm::LogVerbatim("IsoTrack") << "e7x7 " << e7x7P.first << " e9x9 " << e9x9P.first << " e11x11 "
                                       << e11x11P.first << " e15x15 " << e15x15P.first;
          edm::LogVerbatim("IsoTrack") << "e7x7_10Sig " << e7x7_10SigP.first << " e11x11_10Sig " << e11x11_10SigP.first
                                       << " e15x15_10Sig " << e15x15_10SigP.first;
        }

        if (doMC_) {
          // check the energy from SimHits
          spr::eECALSimInfo(
              iEvent, isoCell, geo, caloTopology, pcaloeb, pcaloee, SimTk, SimVtx, pTrack, *associate, 1, 1, simInfo3x3);
          spr::eECALSimInfo(
              iEvent, isoCell, geo, caloTopology, pcaloeb, pcaloee, SimTk, SimVtx, pTrack, *associate, 2, 2, simInfo5x5);
          spr::eECALSimInfo(
              iEvent, isoCell, geo, caloTopology, pcaloeb, pcaloee, SimTk, SimVtx, pTrack, *associate, 3, 3, simInfo7x7);
          spr::eECALSimInfo(
              iEvent, isoCell, geo, caloTopology, pcaloeb, pcaloee, SimTk, SimVtx, pTrack, *associate, 4, 4, simInfo9x9);
          spr::eECALSimInfo(iEvent,
                            isoCell,
                            geo,
                            caloTopology,
                            pcaloeb,
                            pcaloee,
                            SimTk,
                            SimVtx,
                            pTrack,
                            *associate,
                            5,
                            5,
                            simInfo11x11);
          spr::eECALSimInfo(iEvent,
                            isoCell,
                            geo,
                            caloTopology,
                            pcaloeb,
                            pcaloee,
                            SimTk,
                            SimVtx,
                            pTrack,
                            *associate,
                            6,
                            6,
                            simInfo13x13);
          spr::eECALSimInfo(iEvent,
                            isoCell,
                            geo,
                            caloTopology,
                            pcaloeb,
                            pcaloee,
                            SimTk,
                            SimVtx,
                            pTrack,
                            *associate,
                            7,
                            7,
                            simInfo15x15,
                            150.0,
                            false);
          spr::eECALSimInfo(iEvent,
                            isoCell,
                            geo,
                            caloTopology,
                            pcaloeb,
                            pcaloee,
                            SimTk,
                            SimVtx,
                            pTrack,
                            *associate,
                            10,
                            10,
                            simInfo21x21);
          spr::eECALSimInfo(iEvent,
                            isoCell,
                            geo,
                            caloTopology,
                            pcaloeb,
                            pcaloee,
                            SimTk,
                            SimVtx,
                            pTrack,
                            *associate,
                            12,
                            12,
                            simInfo25x25);
          spr::eECALSimInfo(iEvent,
                            isoCell,
                            geo,
                            caloTopology,
                            pcaloeb,
                            pcaloee,
                            SimTk,
                            SimVtx,
                            pTrack,
                            *associate,
                            15,
                            15,
                            simInfo31x31);

          trkEcalEne = spr::eCaloSimInfo(iEvent, geo, pcaloeb, pcaloee, SimTk, SimVtx, pTrack, *associate);
          if (myverbose_ == 1) {
            edm::LogVerbatim("IsoTrack") << "Track momentum " << pt1;
            edm::LogVerbatim("IsoTrack") << "ecal siminfo ";
            edm::LogVerbatim("IsoTrack") << "simInfo3x3: eTotal " << simInfo3x3.eTotal << " eMatched "
                                         << simInfo3x3.eMatched << " eRest " << simInfo3x3.eRest << " eGamma "
                                         << simInfo3x3.eGamma << " eNeutralHad " << simInfo3x3.eNeutralHad
                                         << " eChargedHad " << simInfo3x3.eChargedHad;
            edm::LogVerbatim("IsoTrack") << "simInfo5x5: eTotal " << simInfo5x5.eTotal << " eMatched "
                                         << simInfo5x5.eMatched << " eRest " << simInfo5x5.eRest << " eGamma "
                                         << simInfo5x5.eGamma << " eNeutralHad " << simInfo5x5.eNeutralHad
                                         << " eChargedHad " << simInfo5x5.eChargedHad;
            edm::LogVerbatim("IsoTrack") << "simInfo7x7: eTotal " << simInfo7x7.eTotal << " eMatched "
                                         << simInfo7x7.eMatched << " eRest " << simInfo7x7.eRest << " eGamma "
                                         << simInfo7x7.eGamma << " eNeutralHad " << simInfo7x7.eNeutralHad
                                         << " eChargedHad " << simInfo7x7.eChargedHad;
            edm::LogVerbatim("IsoTrack") << "simInfo9x9: eTotal " << simInfo9x9.eTotal << " eMatched "
                                         << simInfo9x9.eMatched << " eRest " << simInfo9x9.eRest << " eGamma "
                                         << simInfo9x9.eGamma << " eNeutralHad " << simInfo9x9.eNeutralHad
                                         << " eChargedHad " << simInfo9x9.eChargedHad;
            edm::LogVerbatim("IsoTrack") << "simInfo11x11: eTotal " << simInfo11x11.eTotal << " eMatched "
                                         << simInfo11x11.eMatched << " eRest " << simInfo11x11.eRest << " eGamma "
                                         << simInfo11x11.eGamma << " eNeutralHad " << simInfo11x11.eNeutralHad
                                         << " eChargedHad " << simInfo11x11.eChargedHad;
            edm::LogVerbatim("IsoTrack") << "simInfo15x15: eTotal " << simInfo15x15.eTotal << " eMatched "
                                         << simInfo15x15.eMatched << " eRest " << simInfo15x15.eRest << " eGamma "
                                         << simInfo15x15.eGamma << " eNeutralHad " << simInfo15x15.eNeutralHad
                                         << " eChargedHad " << simInfo15x15.eChargedHad;
            edm::LogVerbatim("IsoTrack") << "simInfo31x31: eTotal " << simInfo31x31.eTotal << " eMatched "
                                         << simInfo31x31.eMatched << " eRest " << simInfo31x31.eRest << " eGamma "
                                         << simInfo31x31.eGamma << " eNeutralHad " << simInfo31x31.eNeutralHad
                                         << " eChargedHad " << simInfo31x31.eChargedHad;
            edm::LogVerbatim("IsoTrack") << "trkEcalEne" << trkEcalEne;
          }
        }

        // =======  Get HCAL information
        double hcalScale = 1.0;
        if (std::abs(pTrack->eta()) < 1.4) {
          hcalScale = 120.0;
        } else {
          hcalScale = 135.0;
        }

        double maxNearHcalP3x3 = -1, maxNearHcalP5x5 = -1, maxNearHcalP7x7 = -1;
        maxNearHcalP3x3 = spr::chargeIsolationHcal(nTracks, trkCaloDets, theHBHETopology, 1, 1);
        maxNearHcalP5x5 = spr::chargeIsolationHcal(nTracks, trkCaloDets, theHBHETopology, 2, 2);
        maxNearHcalP7x7 = spr::chargeIsolationHcal(nTracks, trkCaloDets, theHBHETopology, 3, 3);

        double h3x3 = 0, h5x5 = 0, h7x7 = 0;
        double h3x3Sig = 0, h5x5Sig = 0, h7x7Sig = 0;
        double trkHcalEne = 0;
        spr::caloSimInfo hsimInfo3x3, hsimInfo5x5, hsimInfo7x7;

        if (trkDetItr->okHCAL) {
          const DetId ClosestCell(trkDetItr->detIdHCAL);
          // bool includeHO=false, bool algoNew=true, bool debug=false
          h3x3 = spr::eHCALmatrix(
              theHBHETopology, ClosestCell, hbhe, 1, 1, false, true, -100.0, -100.0, -100.0, -100.0, tMinH_, tMaxH_);
          h5x5 = spr::eHCALmatrix(
              theHBHETopology, ClosestCell, hbhe, 2, 2, false, true, -100.0, -100.0, -100.0, -100.0, tMinH_, tMaxH_);
          h7x7 = spr::eHCALmatrix(
              theHBHETopology, ClosestCell, hbhe, 3, 3, false, true, -100.0, -100.0, -100.0, -100.0, tMinH_, tMaxH_);
          h3x3Sig = spr::eHCALmatrix(
              theHBHETopology, ClosestCell, hbhe, 1, 1, false, true, 0.7, 0.8, -100.0, -100.0, tMinH_, tMaxH_);
          h5x5Sig = spr::eHCALmatrix(
              theHBHETopology, ClosestCell, hbhe, 2, 2, false, true, 0.7, 0.8, -100.0, -100.0, tMinH_, tMaxH_);
          h7x7Sig = spr::eHCALmatrix(
              theHBHETopology, ClosestCell, hbhe, 3, 3, false, true, 0.7, 0.8, -100.0, -100.0, tMinH_, tMaxH_);

          if (myverbose_ == 2) {
            edm::LogVerbatim("IsoTrack") << "HCAL 3x3 " << h3x3 << " " << h3x3Sig << " 5x5 " << h5x5 << " " << h5x5Sig
                                         << " 7x7 " << h7x7 << " " << h7x7Sig;
          }

          if (doMC_) {
            spr::eHCALSimInfo(
                iEvent, theHBHETopology, ClosestCell, geo, pcalohh, SimTk, SimVtx, pTrack, *associate, 1, 1, hsimInfo3x3);
            spr::eHCALSimInfo(
                iEvent, theHBHETopology, ClosestCell, geo, pcalohh, SimTk, SimVtx, pTrack, *associate, 2, 2, hsimInfo5x5);
            spr::eHCALSimInfo(iEvent,
                              theHBHETopology,
                              ClosestCell,
                              geo,
                              pcalohh,
                              SimTk,
                              SimVtx,
                              pTrack,
                              *associate,
                              3,
                              3,
                              hsimInfo7x7,
                              150.0,
                              false,
                              false);
            trkHcalEne = spr::eCaloSimInfo(iEvent, geo, pcalohh, SimTk, SimVtx, pTrack, *associate);
            if (myverbose_ == 1) {
              edm::LogVerbatim("IsoTrack") << "Hcal siminfo ";
              edm::LogVerbatim("IsoTrack")
                  << "hsimInfo3x3: eTotal " << hsimInfo3x3.eTotal << " eMatched " << hsimInfo3x3.eMatched << " eRest "
                  << hsimInfo3x3.eRest << " eGamma " << hsimInfo3x3.eGamma << " eNeutralHad " << hsimInfo3x3.eNeutralHad
                  << " eChargedHad " << hsimInfo3x3.eChargedHad;
              edm::LogVerbatim("IsoTrack")
                  << "hsimInfo5x5: eTotal " << hsimInfo5x5.eTotal << " eMatched " << hsimInfo5x5.eMatched << " eRest "
                  << hsimInfo5x5.eRest << " eGamma " << hsimInfo5x5.eGamma << " eNeutralHad " << hsimInfo5x5.eNeutralHad
                  << " eChargedHad " << hsimInfo5x5.eChargedHad;
              edm::LogVerbatim("IsoTrack")
                  << "hsimInfo7x7: eTotal " << hsimInfo7x7.eTotal << " eMatched " << hsimInfo7x7.eMatched << " eRest "
                  << hsimInfo7x7.eRest << " eGamma " << hsimInfo7x7.eGamma << " eNeutralHad " << hsimInfo7x7.eNeutralHad
                  << " eChargedHad " << hsimInfo7x7.eChargedHad;
              edm::LogVerbatim("IsoTrack") << "trkHcalEne " << trkHcalEne;
            }
          }

          // debug the ecal and hcal matrix
          if (myverbose_ == 4) {
            edm::LogVerbatim("IsoTrack") << "Run " << iEvent.id().run() << "  Event " << iEvent.id().event();
            std::vector<std::pair<DetId, double> > v7x7 =
                spr::eHCALmatrixCell(theHBHETopology, ClosestCell, hbhe, 3, 3, false, false);
            double sumv = 0.0;

            for (unsigned int iv = 0; iv < v7x7.size(); iv++) {
              sumv += v7x7[iv].second;
            }
            edm::LogVerbatim("IsoTrack") << "h7x7 " << h7x7 << " v7x7 " << sumv << " in " << v7x7.size();
            for (unsigned int iv = 0; iv < v7x7.size(); iv++) {
              HcalDetId id = v7x7[iv].first;
              edm::LogVerbatim("IsoTrack") << " Cell " << iv << " 0x" << std::hex << v7x7[iv].first() << std::dec << " "
                                           << id << " Energy " << v7x7[iv].second;
            }
          }
        }
        if (doMC_) {
          trkHcalEne = spr::eCaloSimInfo(iEvent, geo, pcalohh, SimTk, SimVtx, pTrack, *associate);
        }

        // ====================================================================================================
        // get diff between track outermost hit position and the propagation point at outermost surface of tracker
        std::pair<math::XYZPoint, double> point2_TK0 = spr::propagateTrackerEnd(pTrack, bField, false);
        math::XYZPoint diff(pTrack->outerPosition().X() - point2_TK0.first.X(),
                            pTrack->outerPosition().Y() - point2_TK0.first.Y(),
                            pTrack->outerPosition().Z() - point2_TK0.first.Z());
        double trackOutPosOutHitDr = diff.R();
        double trackL = point2_TK0.second;
        if (myverbose_ == 5) {
          edm::LogVerbatim("IsoTrack") << " propagted " << point2_TK0.first << " " << point2_TK0.first.eta() << " "
                                       << point2_TK0.first.phi();
          edm::LogVerbatim("IsoTrack") << " outerPosition() " << pTrack->outerPosition() << " "
                                       << pTrack->outerPosition().eta() << " " << pTrack->outerPosition().phi();
          edm::LogVerbatim("IsoTrack") << "diff " << diff << " diffR " << diff.R() << " diffR/L "
                                       << diff.R() / point2_TK0.second;
        }

        for (unsigned int ind = 0; ind < recVtxs->size(); ind++) {
          if (!((*recVtxs)[ind].isFake())) {
            reco::Vertex::trackRef_iterator vtxTrack = (*recVtxs)[ind].tracks_begin();
            if (reco::deltaR(eta1, phi1, (*vtxTrack)->eta(), (*vtxTrack)->phi()) < 0.01)
              t_trackPVIdx.push_back(ind);
            else
              t_trackPVIdx.push_back(-1);
          }
        }

        // Fill the tree Branches here
        t_trackPVIdx.push_back(pVtxTkId);
        t_trackP.push_back(p1);
        t_trackPt.push_back(pt1);
        t_trackEta.push_back(eta1);
        t_trackPhi.push_back(phi1);
        t_trackEcalEta.push_back(etaEcal1);
        t_trackEcalPhi.push_back(phiEcal1);
        t_trackHcalEta.push_back(etaHcal1);
        t_trackHcalPhi.push_back(phiHcal1);
        t_trackDxy.push_back(dxy1);
        t_trackDz.push_back(dz1);
        t_trackDxyBS.push_back(dxybs1);
        t_trackDzBS.push_back(dzbs1);
        t_trackDxyPV.push_back(dxypv1);
        t_trackDzPV.push_back(dzpv1);
        t_trackChiSq.push_back(chisq1);
        t_trackNOuterHits.push_back(nOuterHits);
        t_NLayersCrossed.push_back(nLayersCrossed);

        t_trackHitsTOB.push_back(hitp.stripTOBLayersWithMeasurement());
        t_trackHitsTEC.push_back(hitp.stripTECLayersWithMeasurement());
        t_trackHitInMissTOB.push_back(hitp.stripTOBLayersWithoutMeasurement(reco::HitPattern::MISSING_OUTER_HITS));
        t_trackHitInMissTEC.push_back(hitp.stripTECLayersWithoutMeasurement(reco::HitPattern::MISSING_OUTER_HITS));
        t_trackHitInMissTIB.push_back(hitp.stripTIBLayersWithoutMeasurement(reco::HitPattern::MISSING_INNER_HITS));
        t_trackHitInMissTID.push_back(hitp.stripTIDLayersWithoutMeasurement(reco::HitPattern::MISSING_INNER_HITS));
        t_trackHitInMissTIBTID.push_back(hitp.stripTIBLayersWithoutMeasurement(reco::HitPattern::MISSING_INNER_HITS) +
                                         hitp.stripTIDLayersWithoutMeasurement(reco::HitPattern::MISSING_INNER_HITS));

        t_trackHitOutMissTOB.push_back(hitp.stripTOBLayersWithoutMeasurement(reco::HitPattern::MISSING_OUTER_HITS));
        t_trackHitOutMissTEC.push_back(hitp.stripTECLayersWithoutMeasurement(reco::HitPattern::MISSING_OUTER_HITS));
        t_trackHitOutMissTIB.push_back(hitp.stripTIBLayersWithoutMeasurement(reco::HitPattern::MISSING_INNER_HITS));
        t_trackHitOutMissTID.push_back(hitp.stripTIDLayersWithoutMeasurement(reco::HitPattern::MISSING_INNER_HITS));
        t_trackHitOutMissTOBTEC.push_back(hitp.stripTOBLayersWithoutMeasurement(reco::HitPattern::MISSING_OUTER_HITS) +
                                          hitp.stripTECLayersWithoutMeasurement(reco::HitPattern::MISSING_OUTER_HITS));

        t_trackHitInMeasTOB.push_back(hitp.stripTOBLayersWithMeasurement());
        t_trackHitInMeasTEC.push_back(hitp.stripTECLayersWithMeasurement());
        t_trackHitInMeasTIB.push_back(hitp.stripTIBLayersWithMeasurement());
        t_trackHitInMeasTID.push_back(hitp.stripTIDLayersWithMeasurement());
        t_trackHitOutMeasTOB.push_back(hitp.stripTOBLayersWithMeasurement());
        t_trackHitOutMeasTEC.push_back(hitp.stripTECLayersWithMeasurement());
        t_trackHitOutMeasTIB.push_back(hitp.stripTIBLayersWithMeasurement());
        t_trackHitOutMeasTID.push_back(hitp.stripTIDLayersWithMeasurement());
        t_trackOutPosOutHitDr.push_back(trackOutPosOutHitDr);
        t_trackL.push_back(trackL);

        t_maxNearP31x31.push_back(maxNearP31x31);
        t_maxNearP21x21.push_back(maxNearP21x21);

        t_ecalSpike11x11.push_back(e11x11P.second);
        t_e7x7.push_back(e7x7P.first);
        t_e9x9.push_back(e9x9P.first);
        t_e11x11.push_back(e11x11P.first);
        t_e15x15.push_back(e15x15P.first);

        t_e7x7_10Sig.push_back(e7x7_10SigP.first);
        t_e9x9_10Sig.push_back(e9x9_10SigP.first);
        t_e11x11_10Sig.push_back(e11x11_10SigP.first);
        t_e15x15_10Sig.push_back(e15x15_10SigP.first);
        t_e7x7_15Sig.push_back(e7x7_15SigP.first);
        t_e9x9_15Sig.push_back(e9x9_15SigP.first);
        t_e11x11_15Sig.push_back(e11x11_15SigP.first);
        t_e15x15_15Sig.push_back(e15x15_15SigP.first);
        t_e7x7_20Sig.push_back(e7x7_20SigP.first);
        t_e9x9_20Sig.push_back(e9x9_20SigP.first);
        t_e11x11_20Sig.push_back(e11x11_20SigP.first);
        t_e15x15_20Sig.push_back(e15x15_20SigP.first);
        t_e7x7_25Sig.push_back(e7x7_25SigP.first);
        t_e9x9_25Sig.push_back(e9x9_25SigP.first);
        t_e11x11_25Sig.push_back(e11x11_25SigP.first);
        t_e15x15_25Sig.push_back(e15x15_25SigP.first);
        t_e7x7_30Sig.push_back(e7x7_30SigP.first);
        t_e9x9_30Sig.push_back(e9x9_30SigP.first);
        t_e11x11_30Sig.push_back(e11x11_30SigP.first);
        t_e15x15_30Sig.push_back(e15x15_30SigP.first);

        if (doMC_) {
          t_esim7x7.push_back(simInfo7x7.eTotal);
          t_esim9x9.push_back(simInfo9x9.eTotal);
          t_esim11x11.push_back(simInfo11x11.eTotal);
          t_esim15x15.push_back(simInfo15x15.eTotal);

          t_esim7x7Matched.push_back(simInfo7x7.eMatched);
          t_esim9x9Matched.push_back(simInfo9x9.eMatched);
          t_esim11x11Matched.push_back(simInfo11x11.eMatched);
          t_esim15x15Matched.push_back(simInfo15x15.eMatched);

          t_esim7x7Rest.push_back(simInfo7x7.eRest);
          t_esim9x9Rest.push_back(simInfo9x9.eRest);
          t_esim11x11Rest.push_back(simInfo11x11.eRest);
          t_esim15x15Rest.push_back(simInfo15x15.eRest);

          t_esim7x7Photon.push_back(simInfo7x7.eGamma);
          t_esim9x9Photon.push_back(simInfo9x9.eGamma);
          t_esim11x11Photon.push_back(simInfo11x11.eGamma);
          t_esim15x15Photon.push_back(simInfo15x15.eGamma);

          t_esim7x7NeutHad.push_back(simInfo7x7.eNeutralHad);
          t_esim9x9NeutHad.push_back(simInfo9x9.eNeutralHad);
          t_esim11x11NeutHad.push_back(simInfo11x11.eNeutralHad);
          t_esim15x15NeutHad.push_back(simInfo15x15.eNeutralHad);

          t_esim7x7CharHad.push_back(simInfo7x7.eChargedHad);
          t_esim9x9CharHad.push_back(simInfo9x9.eChargedHad);
          t_esim11x11CharHad.push_back(simInfo11x11.eChargedHad);
          t_esim15x15CharHad.push_back(simInfo15x15.eChargedHad);

          t_trkEcalEne.push_back(trkEcalEne);
          t_simTrackP.push_back(simTrackP);
          t_esimPdgId.push_back(simInfo11x11.pdgMatched);
        }

        t_maxNearHcalP3x3.push_back(maxNearHcalP3x3);
        t_maxNearHcalP5x5.push_back(maxNearHcalP5x5);
        t_maxNearHcalP7x7.push_back(maxNearHcalP7x7);

        t_h3x3.push_back(h3x3);
        t_h5x5.push_back(h5x5);
        t_h7x7.push_back(h7x7);
        t_h3x3Sig.push_back(h3x3Sig);
        t_h5x5Sig.push_back(h5x5Sig);
        t_h7x7Sig.push_back(h7x7Sig);

        t_infoHcal.push_back(trkDetItr->okHCAL);
        if (doMC_) {
          t_trkHcalEne.push_back(hcalScale * trkHcalEne);

          t_hsim3x3.push_back(hcalScale * hsimInfo3x3.eTotal);
          t_hsim5x5.push_back(hcalScale * hsimInfo5x5.eTotal);
          t_hsim7x7.push_back(hcalScale * hsimInfo7x7.eTotal);

          t_hsim3x3Matched.push_back(hcalScale * hsimInfo3x3.eMatched);
          t_hsim5x5Matched.push_back(hcalScale * hsimInfo5x5.eMatched);
          t_hsim7x7Matched.push_back(hcalScale * hsimInfo7x7.eMatched);

          t_hsim3x3Rest.push_back(hcalScale * hsimInfo3x3.eRest);
          t_hsim5x5Rest.push_back(hcalScale * hsimInfo5x5.eRest);
          t_hsim7x7Rest.push_back(hcalScale * hsimInfo7x7.eRest);

          t_hsim3x3Photon.push_back(hcalScale * hsimInfo3x3.eGamma);
          t_hsim5x5Photon.push_back(hcalScale * hsimInfo5x5.eGamma);
          t_hsim7x7Photon.push_back(hcalScale * hsimInfo7x7.eGamma);

          t_hsim3x3NeutHad.push_back(hcalScale * hsimInfo3x3.eNeutralHad);
          t_hsim5x5NeutHad.push_back(hcalScale * hsimInfo5x5.eNeutralHad);
          t_hsim7x7NeutHad.push_back(hcalScale * hsimInfo7x7.eNeutralHad);

          t_hsim3x3CharHad.push_back(hcalScale * hsimInfo3x3.eChargedHad);
          t_hsim5x5CharHad.push_back(hcalScale * hsimInfo5x5.eChargedHad);
          t_hsim7x7CharHad.push_back(hcalScale * hsimInfo7x7.eChargedHad);
        }

      }  // if loosely isolated track
    }    // check p1/eta
  }      // loop over track collection

  if (haveIsoTrack)
    tree_->Fill();
}

// ----- method called once each job just before starting event loop ----
void IsolatedTracksNxN::beginJob() {
  nEventProc_ = 0;
  nbad_ = 0;
  initL1_ = false;
  double tempgen_TH[NPBins + 1] = {
      0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 11.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0};

  for (unsigned int i = 0; i < NPBins + 1; i++)
    genPartPBins[i] = tempgen_TH[i];

  double tempgen_Eta[NEtaBins + 1] = {0.0, 1.131, 1.653, 2.172};

  for (unsigned int i = 0; i < NEtaBins + 1; i++)
    genPartEtaBins[i] = tempgen_Eta[i];

  bookHistograms();
}

// ----- method called once each job just after ending the event loop ----
void IsolatedTracksNxN::endJob() {
  if (L1TriggerAlgoInfo_) {
    std::map<std::pair<unsigned int, std::string>, int>::iterator itr;
    for (itr = l1AlgoMap_.begin(); itr != l1AlgoMap_.end(); itr++) {
      edm::LogVerbatim("IsoTrack") << " ****endjob**** " << (itr->first).first << " " << (itr->first).second << " "
                                   << itr->second;
      int ibin = (itr->first).first;
      TString name((itr->first).second);
      h_L1AlgoNames->GetXaxis()->SetBinLabel(ibin + 1, name);
    }
    edm::LogVerbatim("IsoTrack") << "Number of Events Processed " << nEventProc_;
  }
}

//========================================================================================================

void IsolatedTracksNxN::clearTreeVectors() {
  t_PVx.clear();
  t_PVy.clear();
  t_PVz.clear();
  t_PVisValid.clear();
  t_PVndof.clear();
  t_PVNTracks.clear();
  t_PVNTracksWt.clear();
  t_PVTracksSumPt.clear();
  t_PVTracksSumPtWt.clear();
  t_PVNTracksHP.clear();
  t_PVNTracksHPWt.clear();
  t_PVTracksSumPtHP.clear();
  t_PVTracksSumPtHPWt.clear();

  for (int i = 0; i < 128; i++)
    t_L1Decision[i] = 0;
  t_L1AlgoNames.clear();
  t_L1PreScale.clear();

  t_L1CenJetPt.clear();
  t_L1CenJetEta.clear();
  t_L1CenJetPhi.clear();
  t_L1FwdJetPt.clear();
  t_L1FwdJetEta.clear();
  t_L1FwdJetPhi.clear();
  t_L1TauJetPt.clear();
  t_L1TauJetEta.clear();
  t_L1TauJetPhi.clear();
  t_L1MuonPt.clear();
  t_L1MuonEta.clear();
  t_L1MuonPhi.clear();
  t_L1IsoEMPt.clear();
  t_L1IsoEMEta.clear();
  t_L1IsoEMPhi.clear();
  t_L1NonIsoEMPt.clear();
  t_L1NonIsoEMEta.clear();
  t_L1NonIsoEMPhi.clear();
  t_L1METPt.clear();
  t_L1METEta.clear();
  t_L1METPhi.clear();

  t_jetPt.clear();
  t_jetEta.clear();
  t_jetPhi.clear();
  t_nTrksJetCalo.clear();
  t_nTrksJetVtx.clear();

  t_trackPAll.clear();
  t_trackEtaAll.clear();
  t_trackPhiAll.clear();
  t_trackPdgIdAll.clear();
  t_trackPtAll.clear();
  t_trackDxyAll.clear();
  t_trackDzAll.clear();
  t_trackDxyPVAll.clear();
  t_trackDzPVAll.clear();
  t_trackChiSqAll.clear();

  t_trackP.clear();
  t_trackPt.clear();
  t_trackEta.clear();
  t_trackPhi.clear();
  t_trackEcalEta.clear();
  t_trackEcalPhi.clear();
  t_trackHcalEta.clear();
  t_trackHcalPhi.clear();
  t_NLayersCrossed.clear();
  t_trackNOuterHits.clear();
  t_trackDxy.clear();
  t_trackDxyBS.clear();
  t_trackDz.clear();
  t_trackDzBS.clear();
  t_trackDxyPV.clear();
  t_trackDzPV.clear();
  t_trackChiSq.clear();
  t_trackPVIdx.clear();
  t_trackHitsTOB.clear();
  t_trackHitsTEC.clear();
  t_trackHitInMissTOB.clear();
  t_trackHitInMissTEC.clear();
  t_trackHitInMissTIB.clear();
  t_trackHitInMissTID.clear();
  t_trackHitInMissTIBTID.clear();
  t_trackHitOutMissTOB.clear();
  t_trackHitOutMissTEC.clear();
  t_trackHitOutMissTIB.clear();
  t_trackHitOutMissTID.clear();
  t_trackHitOutMissTOBTEC.clear();
  t_trackHitInMeasTOB.clear();
  t_trackHitInMeasTEC.clear();
  t_trackHitInMeasTIB.clear();
  t_trackHitInMeasTID.clear();
  t_trackHitOutMeasTOB.clear();
  t_trackHitOutMeasTEC.clear();
  t_trackHitOutMeasTIB.clear();
  t_trackHitOutMeasTID.clear();
  t_trackOutPosOutHitDr.clear();
  t_trackL.clear();

  t_maxNearP31x31.clear();
  t_maxNearP21x21.clear();

  t_ecalSpike11x11.clear();
  t_e7x7.clear();
  t_e9x9.clear();
  t_e11x11.clear();
  t_e15x15.clear();

  t_e7x7_10Sig.clear();
  t_e9x9_10Sig.clear();
  t_e11x11_10Sig.clear();
  t_e15x15_10Sig.clear();
  t_e7x7_15Sig.clear();
  t_e9x9_15Sig.clear();
  t_e11x11_15Sig.clear();
  t_e15x15_15Sig.clear();
  t_e7x7_20Sig.clear();
  t_e9x9_20Sig.clear();
  t_e11x11_20Sig.clear();
  t_e15x15_20Sig.clear();
  t_e7x7_25Sig.clear();
  t_e9x9_25Sig.clear();
  t_e11x11_25Sig.clear();
  t_e15x15_25Sig.clear();
  t_e7x7_30Sig.clear();
  t_e9x9_30Sig.clear();
  t_e11x11_30Sig.clear();
  t_e15x15_30Sig.clear();

  if (doMC_) {
    t_simTrackP.clear();
    t_esimPdgId.clear();
    t_trkEcalEne.clear();

    t_esim7x7.clear();
    t_esim9x9.clear();
    t_esim11x11.clear();
    t_esim15x15.clear();

    t_esim7x7Matched.clear();
    t_esim9x9Matched.clear();
    t_esim11x11Matched.clear();
    t_esim15x15Matched.clear();

    t_esim7x7Rest.clear();
    t_esim9x9Rest.clear();
    t_esim11x11Rest.clear();
    t_esim15x15Rest.clear();

    t_esim7x7Photon.clear();
    t_esim9x9Photon.clear();
    t_esim11x11Photon.clear();
    t_esim15x15Photon.clear();

    t_esim7x7NeutHad.clear();
    t_esim9x9NeutHad.clear();
    t_esim11x11NeutHad.clear();
    t_esim15x15NeutHad.clear();

    t_esim7x7CharHad.clear();
    t_esim9x9CharHad.clear();
    t_esim11x11CharHad.clear();
    t_esim15x15CharHad.clear();
  }

  t_maxNearHcalP3x3.clear();
  t_maxNearHcalP5x5.clear();
  t_maxNearHcalP7x7.clear();

  t_h3x3.clear();
  t_h5x5.clear();
  t_h7x7.clear();
  t_h3x3Sig.clear();
  t_h5x5Sig.clear();
  t_h7x7Sig.clear();

  t_infoHcal.clear();

  if (doMC_) {
    t_trkHcalEne.clear();

    t_hsim3x3.clear();
    t_hsim5x5.clear();
    t_hsim7x7.clear();
    t_hsim3x3Matched.clear();
    t_hsim5x5Matched.clear();
    t_hsim7x7Matched.clear();
    t_hsim3x3Rest.clear();
    t_hsim5x5Rest.clear();
    t_hsim7x7Rest.clear();
    t_hsim3x3Photon.clear();
    t_hsim5x5Photon.clear();
    t_hsim7x7Photon.clear();
    t_hsim3x3NeutHad.clear();
    t_hsim5x5NeutHad.clear();
    t_hsim7x7NeutHad.clear();
    t_hsim3x3CharHad.clear();
    t_hsim5x5CharHad.clear();
    t_hsim7x7CharHad.clear();
  }
}

void IsolatedTracksNxN::bookHistograms() {
  char hname[100], htit[100];

  edm::Service<TFileService> fs;
  TFileDirectory dir = fs->mkdir("nearMaxTrackP");

  for (unsigned int ieta = 0; ieta < NEtaBins; ieta++) {
    double lowEta = -5.0, highEta = 5.0;
    lowEta = genPartEtaBins[ieta];
    highEta = genPartEtaBins[ieta + 1];

    for (unsigned int ipt = 0; ipt < NPBins; ipt++) {
      double lowP = 0.0, highP = 300.0;
      lowP = genPartPBins[ipt];
      highP = genPartPBins[ipt + 1];
      sprintf(hname, "h_maxNearP31x31_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit, "maxNearP in 31x31 (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP);
      h_maxNearP31x31[ipt][ieta] = dir.make<TH1F>(hname, htit, 220, -2.0, 20.0);
      h_maxNearP31x31[ipt][ieta]->Sumw2();
      sprintf(hname, "h_maxNearP25x25_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit, "maxNearP in 25x25 (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP);
      h_maxNearP25x25[ipt][ieta] = dir.make<TH1F>(hname, htit, 220, -2.0, 20.0);
      h_maxNearP25x25[ipt][ieta]->Sumw2();
      sprintf(hname, "h_maxNearP21x21_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit, "maxNearP in 21x21 (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP);
      h_maxNearP21x21[ipt][ieta] = dir.make<TH1F>(hname, htit, 220, -2.0, 20.0);
      h_maxNearP21x21[ipt][ieta]->Sumw2();
      sprintf(hname, "h_maxNearP15x15_ptBin%i_etaBin%i", ipt, ieta);
      sprintf(htit, "maxNearP in 15x15 (%3.2f<|#eta|<%3.2f), (%2.0f<trkP<%3.0f)", lowEta, highEta, lowP, highP);
      h_maxNearP15x15[ipt][ieta] = dir.make<TH1F>(hname, htit, 220, -2.0, 20.0);
      h_maxNearP15x15[ipt][ieta]->Sumw2();
    }
  }

  h_L1AlgoNames = fs->make<TH1I>("h_L1AlgoNames", "h_L1AlgoNames:Bin Labels", 128, -0.5, 127.5);

  // Reconstructed Tracks

  h_PVTracksWt = fs->make<TH1F>("h_PVTracksWt", "h_PVTracksWt", 600, -0.1, 1.1);

  h_nTracks = fs->make<TH1F>("h_nTracks", "h_nTracks", 1000, -0.5, 999.5);

  sprintf(hname, "h_recEtaPt_0");
  sprintf(htit, "h_recEtaPt (all tracks Eta vs pT)");
  h_recEtaPt_0 = fs->make<TH2F>(hname, htit, 30, -3.0, 3.0, 15, genPartPBins);

  sprintf(hname, "h_recEtaP_0");
  sprintf(htit, "h_recEtaP (all tracks Eta vs pT)");
  h_recEtaP_0 = fs->make<TH2F>(hname, htit, 30, -3.0, 3.0, 15, genPartPBins);

  h_recPt_0 = fs->make<TH1F>("h_recPt_0", "Pt (all tracks)", 15, genPartPBins);
  h_recP_0 = fs->make<TH1F>("h_recP_0", "P  (all tracks)", 15, genPartPBins);
  h_recEta_0 = fs->make<TH1F>("h_recEta_0", "Eta (all tracks)", 60, -3.0, 3.0);
  h_recPhi_0 = fs->make<TH1F>("h_recPhi_0", "Phi (all tracks)", 100, -3.2, 3.2);
  //-------------------------
  sprintf(hname, "h_recEtaPt_1");
  sprintf(htit, "h_recEtaPt (all good tracks Eta vs pT)");
  h_recEtaPt_1 = fs->make<TH2F>(hname, htit, 30, -3.0, 3.0, 15, genPartPBins);

  sprintf(hname, "h_recEtaP_1");
  sprintf(htit, "h_recEtaP (all good tracks Eta vs pT)");
  h_recEtaP_1 = fs->make<TH2F>(hname, htit, 30, -3.0, 3.0, 15, genPartPBins);

  h_recPt_1 = fs->make<TH1F>("h_recPt_1", "Pt (all good tracks)", 15, genPartPBins);
  h_recP_1 = fs->make<TH1F>("h_recP_1", "P  (all good tracks)", 15, genPartPBins);
  h_recEta_1 = fs->make<TH1F>("h_recEta_1", "Eta (all good tracks)", 60, -3.0, 3.0);
  h_recPhi_1 = fs->make<TH1F>("h_recPhi_1", "Phi (all good tracks)", 100, -3.2, 3.2);
  //-------------------------
  sprintf(hname, "h_recEtaPt_2");
  sprintf(htit, "h_recEtaPt (charge isolation Eta vs pT)");
  h_recEtaPt_2 = fs->make<TH2F>(hname, htit, 30, -3.0, 3.0, 15, genPartPBins);

  sprintf(hname, "h_recEtaP_2");
  sprintf(htit, "h_recEtaP (charge isolation Eta vs pT)");
  h_recEtaP_2 = fs->make<TH2F>(hname, htit, 30, -3.0, 3.0, 15, genPartPBins);

  h_recPt_2 = fs->make<TH1F>("h_recPt_2", "Pt (charge isolation)", 15, genPartPBins);
  h_recP_2 = fs->make<TH1F>("h_recP_2", "P  (charge isolation)", 15, genPartPBins);
  h_recEta_2 = fs->make<TH1F>("h_recEta_2", "Eta (charge isolation)", 60, -3.0, 3.0);
  h_recPhi_2 = fs->make<TH1F>("h_recPhi_2", "Phi (charge isolation)", 100, -3.2, 3.2);

  tree_ = fs->make<TTree>("tree", "tree");
  tree_->SetAutoSave(10000);

  tree_->Branch("t_EvtNo", &t_EvtNo, "t_EvtNo/I");
  tree_->Branch("t_RunNo", &t_RunNo, "t_RunNo/I");
  tree_->Branch("t_Lumi", &t_Lumi, "t_Lumi/I");
  tree_->Branch("t_Bunch", &t_Bunch, "t_Bunch/I");

  //----- L1Trigger information
  for (int i = 0; i < 128; i++)
    t_L1Decision[i] = 0;

  tree_->Branch("t_L1Decision", t_L1Decision, "t_L1Decision[128]/I");
  tree_->Branch("t_L1AlgoNames", &t_L1AlgoNames);
  tree_->Branch("t_L1PreScale", &t_L1PreScale);
  tree_->Branch("t_L1CenJetPt", &t_L1CenJetPt);
  tree_->Branch("t_L1CenJetEta", &t_L1CenJetEta);
  tree_->Branch("t_L1CenJetPhi", &t_L1CenJetPhi);
  tree_->Branch("t_L1FwdJetPt", &t_L1FwdJetPt);
  tree_->Branch("t_L1FwdJetEta", &t_L1FwdJetEta);
  tree_->Branch("t_L1FwdJetPhi", &t_L1FwdJetPhi);
  tree_->Branch("t_L1TauJetPt", &t_L1TauJetPt);
  tree_->Branch("t_L1TauJetEta", &t_L1TauJetEta);
  tree_->Branch("t_L1TauJetPhi", &t_L1TauJetPhi);
  tree_->Branch("t_L1MuonPt", &t_L1MuonPt);
  tree_->Branch("t_L1MuonEta", &t_L1MuonEta);
  tree_->Branch("t_L1MuonPhi", &t_L1MuonPhi);
  tree_->Branch("t_L1IsoEMPt", &t_L1IsoEMPt);
  tree_->Branch("t_L1IsoEMEta", &t_L1IsoEMEta);
  tree_->Branch("t_L1IsoEMPhi", &t_L1IsoEMPhi);
  tree_->Branch("t_L1NonIsoEMPt", &t_L1NonIsoEMPt);
  tree_->Branch("t_L1NonIsoEMEta", &t_L1NonIsoEMEta);
  tree_->Branch("t_L1NonIsoEMPhi", &t_L1NonIsoEMPhi);
  tree_->Branch("t_L1METPt", &t_L1METPt);
  tree_->Branch("t_L1METEta", &t_L1METEta);
  tree_->Branch("t_L1METPhi", &t_L1METPhi);

  tree_->Branch("t_jetPt", &t_jetPt);
  tree_->Branch("t_jetEta", &t_jetEta);
  tree_->Branch("t_jetPhi", &t_jetPhi);
  tree_->Branch("t_nTrksJetCalo", &t_nTrksJetCalo);
  tree_->Branch("t_nTrksJetVtx", &t_nTrksJetVtx);

  tree_->Branch("t_trackPAll", &t_trackPAll);
  tree_->Branch("t_trackPhiAll", &t_trackPhiAll);
  tree_->Branch("t_trackEtaAll", &t_trackEtaAll);
  tree_->Branch("t_trackPtAll", &t_trackPtAll);
  tree_->Branch("t_trackDxyAll", &t_trackDxyAll);
  tree_->Branch("t_trackDzAll", &t_trackDzAll);
  tree_->Branch("t_trackDxyPVAll", &t_trackDxyPVAll);
  tree_->Branch("t_trackDzPVAll", &t_trackDzPVAll);
  tree_->Branch("t_trackChiSqAll", &t_trackChiSqAll);
  //tree_->Branch("t_trackPdgIdAll",     &t_trackPdgIdAll);

  tree_->Branch("t_trackP", &t_trackP);
  tree_->Branch("t_trackPt", &t_trackPt);
  tree_->Branch("t_trackEta", &t_trackEta);
  tree_->Branch("t_trackPhi", &t_trackPhi);
  tree_->Branch("t_trackEcalEta", &t_trackEcalEta);
  tree_->Branch("t_trackEcalPhi", &t_trackEcalPhi);
  tree_->Branch("t_trackHcalEta", &t_trackHcalEta);
  tree_->Branch("t_trackHcalPhi", &t_trackHcalPhi);

  tree_->Branch("t_trackNOuterHits", &t_trackNOuterHits);
  tree_->Branch("t_NLayersCrossed", &t_NLayersCrossed);
  tree_->Branch("t_trackHitsTOB", &t_trackHitsTOB);
  tree_->Branch("t_trackHitsTEC", &t_trackHitsTEC);
  tree_->Branch("t_trackHitInMissTOB", &t_trackHitInMissTOB);
  tree_->Branch("t_trackHitInMissTEC", &t_trackHitInMissTEC);
  tree_->Branch("t_trackHitInMissTIB", &t_trackHitInMissTIB);
  tree_->Branch("t_trackHitInMissTID", &t_trackHitInMissTID);
  tree_->Branch("t_trackHitInMissTIBTID", &t_trackHitInMissTIBTID);
  tree_->Branch("t_trackHitOutMissTOB", &t_trackHitOutMissTOB);
  tree_->Branch("t_trackHitOutMissTEC", &t_trackHitOutMissTEC);
  tree_->Branch("t_trackHitOutMissTIB", &t_trackHitOutMissTIB);
  tree_->Branch("t_trackHitOutMissTID", &t_trackHitOutMissTID);
  tree_->Branch("t_trackHitOutMissTOBTEC", &t_trackHitOutMissTOBTEC);
  tree_->Branch("t_trackHitInMeasTOB", &t_trackHitInMeasTOB);
  tree_->Branch("t_trackHitInMeasTEC", &t_trackHitInMeasTEC);
  tree_->Branch("t_trackHitInMeasTIB", &t_trackHitInMeasTIB);
  tree_->Branch("t_trackHitInMeasTID", &t_trackHitInMeasTID);
  tree_->Branch("t_trackHitOutMeasTOB", &t_trackHitOutMeasTOB);
  tree_->Branch("t_trackHitOutMeasTEC", &t_trackHitOutMeasTEC);
  tree_->Branch("t_trackHitOutMeasTIB", &t_trackHitOutMeasTIB);
  tree_->Branch("t_trackHitOutMeasTID", &t_trackHitOutMeasTID);
  tree_->Branch("t_trackOutPosOutHitDr", &t_trackOutPosOutHitDr);
  tree_->Branch("t_trackL", &t_trackL);

  tree_->Branch("t_trackDxy", &t_trackDxy);
  tree_->Branch("t_trackDxyBS", &t_trackDxyBS);
  tree_->Branch("t_trackDz", &t_trackDz);
  tree_->Branch("t_trackDzBS", &t_trackDzBS);
  tree_->Branch("t_trackDxyPV", &t_trackDxyPV);
  tree_->Branch("t_trackDzPV", &t_trackDzPV);
  tree_->Branch("t_trackChiSq", &t_trackChiSq);
  tree_->Branch("t_trackPVIdx", &t_trackPVIdx);

  tree_->Branch("t_maxNearP31x31", &t_maxNearP31x31);
  tree_->Branch("t_maxNearP21x21", &t_maxNearP21x21);

  tree_->Branch("t_ecalSpike11x11", &t_ecalSpike11x11);
  tree_->Branch("t_e7x7", &t_e7x7);
  tree_->Branch("t_e9x9", &t_e9x9);
  tree_->Branch("t_e11x11", &t_e11x11);
  tree_->Branch("t_e15x15", &t_e15x15);

  tree_->Branch("t_e7x7_10Sig", &t_e7x7_10Sig);
  tree_->Branch("t_e9x9_10Sig", &t_e9x9_10Sig);
  tree_->Branch("t_e11x11_10Sig", &t_e11x11_10Sig);
  tree_->Branch("t_e15x15_10Sig", &t_e15x15_10Sig);
  tree_->Branch("t_e7x7_15Sig", &t_e7x7_15Sig);
  tree_->Branch("t_e9x9_15Sig", &t_e9x9_15Sig);
  tree_->Branch("t_e11x11_15Sig", &t_e11x11_15Sig);
  tree_->Branch("t_e15x15_15Sig", &t_e15x15_15Sig);
  tree_->Branch("t_e7x7_20Sig", &t_e7x7_20Sig);
  tree_->Branch("t_e9x9_20Sig", &t_e9x9_20Sig);
  tree_->Branch("t_e11x11_20Sig", &t_e11x11_20Sig);
  tree_->Branch("t_e15x15_20Sig", &t_e15x15_20Sig);
  tree_->Branch("t_e7x7_25Sig", &t_e7x7_25Sig);
  tree_->Branch("t_e9x9_25Sig", &t_e9x9_25Sig);
  tree_->Branch("t_e11x11_25Sig", &t_e11x11_25Sig);
  tree_->Branch("t_e15x15_25Sig", &t_e15x15_25Sig);
  tree_->Branch("t_e7x7_30Sig", &t_e7x7_30Sig);
  tree_->Branch("t_e9x9_30Sig", &t_e9x9_30Sig);
  tree_->Branch("t_e11x11_30Sig", &t_e11x11_30Sig);
  tree_->Branch("t_e15x15_30Sig", &t_e15x15_30Sig);

  if (doMC_) {
    tree_->Branch("t_esim7x7", &t_esim7x7);
    tree_->Branch("t_esim9x9", &t_esim9x9);
    tree_->Branch("t_esim11x11", &t_esim11x11);
    tree_->Branch("t_esim15x15", &t_esim15x15);

    tree_->Branch("t_esim7x7Matched", &t_esim7x7Matched);
    tree_->Branch("t_esim9x9Matched", &t_esim9x9Matched);
    tree_->Branch("t_esim11x11Matched", &t_esim11x11Matched);
    tree_->Branch("t_esim15x15Matched", &t_esim15x15Matched);

    tree_->Branch("t_esim7x7Rest", &t_esim7x7Rest);
    tree_->Branch("t_esim9x9Rest", &t_esim9x9Rest);
    tree_->Branch("t_esim11x11Rest", &t_esim11x11Rest);
    tree_->Branch("t_esim15x15Rest", &t_esim15x15Rest);

    tree_->Branch("t_esim7x7Photon", &t_esim7x7Photon);
    tree_->Branch("t_esim9x9Photon", &t_esim9x9Photon);
    tree_->Branch("t_esim11x11Photon", &t_esim11x11Photon);
    tree_->Branch("t_esim15x15Photon", &t_esim15x15Photon);

    tree_->Branch("t_esim7x7NeutHad", &t_esim7x7NeutHad);
    tree_->Branch("t_esim9x9NeutHad", &t_esim9x9NeutHad);
    tree_->Branch("t_esim11x11NeutHad", &t_esim11x11NeutHad);
    tree_->Branch("t_esim15x15NeutHad", &t_esim15x15NeutHad);

    tree_->Branch("t_esim7x7CharHad", &t_esim7x7CharHad);
    tree_->Branch("t_esim9x9CharHad", &t_esim9x9CharHad);
    tree_->Branch("t_esim11x11CharHad", &t_esim11x11CharHad);
    tree_->Branch("t_esim15x15CharHad", &t_esim15x15CharHad);

    tree_->Branch("t_trkEcalEne", &t_trkEcalEne);
    tree_->Branch("t_simTrackP", &t_simTrackP);
    tree_->Branch("t_esimPdgId", &t_esimPdgId);
  }

  tree_->Branch("t_maxNearHcalP3x3", &t_maxNearHcalP3x3);
  tree_->Branch("t_maxNearHcalP5x5", &t_maxNearHcalP5x5);
  tree_->Branch("t_maxNearHcalP7x7", &t_maxNearHcalP7x7);
  tree_->Branch("t_h3x3", &t_h3x3);
  tree_->Branch("t_h5x5", &t_h5x5);
  tree_->Branch("t_h7x7", &t_h7x7);
  tree_->Branch("t_h3x3Sig", &t_h3x3Sig);
  tree_->Branch("t_h5x5Sig", &t_h5x5Sig);
  tree_->Branch("t_h7x7Sig", &t_h7x7Sig);
  tree_->Branch("t_infoHcal", &t_infoHcal);

  if (doMC_) {
    tree_->Branch("t_trkHcalEne", &t_trkHcalEne);
    tree_->Branch("t_hsim3x3", &t_hsim3x3);
    tree_->Branch("t_hsim5x5", &t_hsim5x5);
    tree_->Branch("t_hsim7x7", &t_hsim7x7);
    tree_->Branch("t_hsim3x3Matched", &t_hsim3x3Matched);
    tree_->Branch("t_hsim5x5Matched", &t_hsim5x5Matched);
    tree_->Branch("t_hsim7x7Matched", &t_hsim7x7Matched);
    tree_->Branch("t_hsim3x3Rest", &t_hsim3x3Rest);
    tree_->Branch("t_hsim5x5Rest", &t_hsim5x5Rest);
    tree_->Branch("t_hsim7x7Rest", &t_hsim7x7Rest);
    tree_->Branch("t_hsim3x3Photon", &t_hsim3x3Photon);
    tree_->Branch("t_hsim5x5Photon", &t_hsim5x5Photon);
    tree_->Branch("t_hsim7x7Photon", &t_hsim7x7Photon);
    tree_->Branch("t_hsim3x3NeutHad", &t_hsim3x3NeutHad);
    tree_->Branch("t_hsim5x5NeutHad", &t_hsim5x5NeutHad);
    tree_->Branch("t_hsim7x7NeutHad", &t_hsim7x7NeutHad);
    tree_->Branch("t_hsim3x3CharHad", &t_hsim3x3CharHad);
    tree_->Branch("t_hsim5x5CharHad", &t_hsim5x5CharHad);
    tree_->Branch("t_hsim7x7CharHad", &t_hsim7x7CharHad);
  }
  tree_->Branch("t_nTracks", &t_nTracks, "t_nTracks/I");
}

void IsolatedTracksNxN::printTrack(const reco::Track *pTrack) {
  std::string theTrackQuality = "highPurity";
  reco::TrackBase::TrackQuality trackQuality_ = reco::TrackBase::qualityByName(theTrackQuality);

  edm::LogVerbatim("IsoTrack") << " Reference Point " << pTrack->referencePoint() << "\n TrackMmentum "
                               << pTrack->momentum() << " (pt,eta,phi)(" << pTrack->pt() << "," << pTrack->eta() << ","
                               << pTrack->phi() << ")"
                               << " p " << pTrack->p() << "\n Normalized chi2 " << pTrack->normalizedChi2()
                               << "  charge " << pTrack->charge() << " qoverp() " << pTrack->qoverp() << "+-"
                               << pTrack->qoverpError() << " d0 " << pTrack->d0() << "\n NValidHits "
                               << pTrack->numberOfValidHits() << "  NLostHits " << pTrack->numberOfLostHits()
                               << " TrackQuality " << pTrack->qualityName(trackQuality_) << " "
                               << pTrack->quality(trackQuality_);

  if (printTrkHitPattern_) {
    const reco::HitPattern &p = pTrack->hitPattern();

    std::ostringstream st1;
    st1 << "default ";
    for (int i = 0; i < p.numberOfAllHits(reco::HitPattern::TRACK_HITS); i++) {
      p.printHitPattern(reco::HitPattern::TRACK_HITS, i, st1);
    }
    edm::LogVerbatim("IsoTrack") << st1.str();
    std::ostringstream st2;
    st2 << "trackerMissingInnerHits() ";
    for (int i = 0; i < p.numberOfAllHits(reco::HitPattern::MISSING_INNER_HITS); i++) {
      p.printHitPattern(reco::HitPattern::MISSING_INNER_HITS, i, st2);
    }
    edm::LogVerbatim("IsoTrack") << st2.str();
    std::ostringstream st3;
    st3 << "trackerMissingOuterHits() ";
    for (int i = 0; i < p.numberOfAllHits(reco::HitPattern::MISSING_OUTER_HITS); i++) {
      p.printHitPattern(reco::HitPattern::MISSING_OUTER_HITS, i, st3);
    }
    edm::LogVerbatim("IsoTrack") << st3.str();

    edm::LogVerbatim("IsoTrack") << "\n \t trackerLayersWithMeasurement() " << p.trackerLayersWithMeasurement()
                                 << "\n \t pixelLayersWithMeasurement() " << p.pixelLayersWithMeasurement()
                                 << "\n \t stripLayersWithMeasurement() " << p.stripLayersWithMeasurement()
                                 << "\n \t pixelBarrelLayersWithMeasurement() " << p.pixelBarrelLayersWithMeasurement()
                                 << "\n \t pixelEndcapLayersWithMeasurement() " << p.pixelEndcapLayersWithMeasurement()
                                 << "\n \t stripTIBLayersWithMeasurement() " << p.stripTIBLayersWithMeasurement()
                                 << "\n \t stripTIDLayersWithMeasurement() " << p.stripTIDLayersWithMeasurement()
                                 << "\n \t stripTOBLayersWithMeasurement() " << p.stripTOBLayersWithMeasurement()
                                 << "\n \t stripTECLayersWithMeasurement() " << p.stripTECLayersWithMeasurement();
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(IsolatedTracksNxN);
