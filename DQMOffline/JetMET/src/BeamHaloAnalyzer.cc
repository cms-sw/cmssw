#include "DQMOffline/JetMET/interface/BeamHaloAnalyzer.h"
//author : Ronny Remington, University of Florida
//date : 11/11/09

using namespace edm;
using namespace reco;

int Phi_To_iPhi(float phi) {
  phi = phi < 0 ? phi + 2. * TMath::Pi() : phi;
  float phi_degrees = phi * (360.) / (2. * TMath::Pi());
  int iPhi = (int)((phi_degrees / 5.) + 1.);

  return iPhi < 73 ? iPhi : 73;
}

BeamHaloAnalyzer::BeamHaloAnalyzer(const edm::ParameterSet& iConfig) {
  OutputFileName = iConfig.getParameter<std::string>("OutputFile");
  TextFileName = iConfig.getParameter<std::string>("TextFile");

  if (!TextFileName.empty())
    out = new std::ofstream(TextFileName.c_str());

  if (iConfig.exists(
          "StandardDQM"))  // If StandardDQM == true , coarse binning is used on selected (important) histograms
    StandardDQM = iConfig.getParameter<bool>("StandardDQM");
  else
    StandardDQM = false;

  //Get Input Tags
  //Digi Level
  IT_L1MuGMTReadout = iConfig.getParameter<edm::InputTag>("L1MuGMTReadoutLabel");

  //RecHit Level
  IT_CSCRecHit = consumes<CSCRecHit2DCollection>(iConfig.getParameter<edm::InputTag>("CSCRecHitLabel"));
  IT_EBRecHit = consumes<EBRecHitCollection>(iConfig.getParameter<edm::InputTag>("EBRecHitLabel"));
  IT_EERecHit = consumes<EERecHitCollection>(iConfig.getParameter<edm::InputTag>("EERecHitLabel"));
  IT_ESRecHit = consumes<ESRecHitCollection>(iConfig.getParameter<edm::InputTag>("ESRecHitLabel"));
  IT_HBHERecHit = consumes<HBHERecHitCollection>(iConfig.getParameter<edm::InputTag>("HBHERecHitLabel"));
  IT_HFRecHit = consumes<HFRecHitCollection>(iConfig.getParameter<edm::InputTag>("HFRecHitLabel"));
  IT_HORecHit = consumes<HORecHitCollection>(iConfig.getParameter<edm::InputTag>("HORecHitLabel"));

  //Higher Level Reco
  IT_CSCSegment = consumes<CSCSegmentCollection>(iConfig.getParameter<edm::InputTag>("CSCSegmentLabel"));
  IT_CosmicStandAloneMuon =
      consumes<reco::MuonCollection>(iConfig.getParameter<edm::InputTag>("CosmicStandAloneMuonLabel"));
  IT_BeamHaloMuon = consumes<reco::MuonCollection>(iConfig.getParameter<edm::InputTag>("BeamHaloMuonLabel"));
  IT_CollisionMuon = consumes<reco::MuonCollection>(iConfig.getParameter<edm::InputTag>("CollisionMuonLabel"));
  IT_CollisionStandAloneMuon =
      consumes<reco::MuonCollection>(iConfig.getParameter<edm::InputTag>("CollisionStandAloneMuonLabel"));
  IT_met = consumes<reco::CaloMETCollection>(iConfig.getParameter<edm::InputTag>("metLabel"));
  IT_CaloTower = consumes<edm::View<reco::Candidate> >(iConfig.getParameter<edm::InputTag>("CaloTowerLabel"));
  IT_SuperCluster = consumes<SuperClusterCollection>(iConfig.getParameter<edm::InputTag>("SuperClusterLabel"));
  IT_Photon = consumes<reco::PhotonCollection>(iConfig.getParameter<edm::InputTag>("PhotonLabel"));

  //Halo Data
  IT_CSCHaloData = consumes<reco::CSCHaloData>(iConfig.getParameter<edm::InputTag>("CSCHaloDataLabel"));
  IT_EcalHaloData = consumes<reco::EcalHaloData>(iConfig.getParameter<edm::InputTag>("EcalHaloDataLabel"));
  IT_HcalHaloData = consumes<reco::HcalHaloData>(iConfig.getParameter<edm::InputTag>("HcalHaloDataLabel"));
  IT_GlobalHaloData = consumes<reco::GlobalHaloData>(iConfig.getParameter<edm::InputTag>("GlobalHaloDataLabel"));
  IT_BeamHaloSummary = consumes<BeamHaloSummary>(iConfig.getParameter<edm::InputTag>("BeamHaloSummaryLabel"));

  cscGeomToken_ = esConsumes();

  edm::InputTag CosmicSAMuonLabel = iConfig.getParameter<edm::InputTag>("CosmicStandAloneMuonLabel");
  IT_CSCTimeMapToken = consumes<reco::MuonTimeExtraMap>(edm::InputTag(CosmicSAMuonLabel.label(), std::string("csc")));
  FolderName = iConfig.getParameter<std::string>("folderName");
  DumpMET = iConfig.getParameter<double>("DumpMET");

  //Muon to Segment Matching
  edm::ParameterSet matchParameters = iConfig.getParameter<edm::ParameterSet>("MatchParameters");
  edm::ConsumesCollector iC = consumesCollector();
  TheMatcher = new MuonSegmentMatcher(matchParameters, iC);
}

void BeamHaloAnalyzer::bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const&) {
  // EcalHaloData
  ibooker.setCurrentFolder(FolderName + "/EcalHaloData");
  if (StandardDQM) {
    hEcalHaloData_PhiWedgeMultiplicity = ibooker.book1D("EcalHaloData_PhiWedgeMultiplicity", "", 20, -0.5, 19.5);
    hEcalHaloData_PhiWedgeConstituents = ibooker.book1D("EcalHaloData_PhiWedgeConstituents", "", 20, -0.5, 19.5);
    //	hEcalHaloData_PhiWedgeiPhi         = ibooker.book1D("EcalHaloData_PhiWedgeiPhi","", 360, 0.5, 360.5) ;
    hEcalHaloData_PhiWedgeZDirectionConfidence =
        ibooker.book1D("EcalHaloData_ZDirectionConfidence", "", 120, -1.2, 1.2);
    hEcalHaloData_SuperClusterShowerShapes =
        ibooker.book2D("EcalHaloData_SuperClusterShowerShapes", "", 30, 0, 3.2, 25, 0.0, 2.0);
    hEcalHaloData_SuperClusterEnergy = ibooker.book1D("EcalHaloData_SuperClusterEnergy", "", 50, -0.5, 99.5);
    hEcalHaloData_SuperClusterNHits = ibooker.book1D("EcalHaloData_SuperClusterNHits", "", 20, -0.5, 19.5);
  } else {
    hEcalHaloData_PhiWedgeMultiplicity = ibooker.book1D("EcalHaloData_PhiWedgeMultiplicity", "", 20, -0.5, 19.5);
    hEcalHaloData_PhiWedgeEnergy = ibooker.book1D("EcalHaloData_PhiWedgeEnergy", "", 50, -0.5, 199.5);
    hEcalHaloData_PhiWedgeConstituents = ibooker.book1D("EcalHaloData_PhiWedgeConstituents", "", 20, -0.5, 19.5);
    hEcalHaloData_PhiWedgeMinTime = ibooker.book1D("EcalHaloData_PhiWedgeMinTime", "", 100, -225.0, 225.0);
    hEcalHaloData_PhiWedgeMaxTime = ibooker.book1D("EcalHaloData_PhiWedgeMaxTime", "", 100, -225.0, 225.0);
    hEcalHaloData_PhiWedgeiPhi = ibooker.book1D("EcalHaloData_PhiWedgeiPhi", "", 360, 0.5, 360.5);
    hEcalHaloData_PhiWedgePlusZDirectionConfidence =
        ibooker.book1D("EcalHaloData_PlusZDirectionConfidence", "", 50, 0., 1.0);
    hEcalHaloData_PhiWedgeZDirectionConfidence =
        ibooker.book1D("EcalHaloData_ZDirectionConfidence", "", 120, -1.2, 1.2);
    hEcalHaloData_PhiWedgeMinVsMaxTime =
        ibooker.book2D("EcalHaloData_PhiWedgeMinVsMaxTime", "", 50, -100.0, 100.0, 50, -100.0, 100.0);
    hEcalHaloData_SuperClusterShowerShapes =
        ibooker.book2D("EcalHaloData_SuperClusterShowerShapes", "", 30, 0, 3.2, 25, 0.0, 2.0);
    hEcalHaloData_SuperClusterEnergy = ibooker.book1D("EcalHaloData_SuperClusterEnergy", "", 100, -0.5, 99.5);
    hEcalHaloData_SuperClusterNHits = ibooker.book1D("EcalHaloData_SuperClusterNHits", "", 20, -0.5, 19.5);
    hEcalHaloData_SuperClusterPhiVsEta =
        ibooker.book2D("EcalHaloData_SuperClusterPhiVsEta", "", 60, -3.0, 3.0, 60, -3.2, 3.2);
  }

  // HcalHaloData
  ibooker.setCurrentFolder(FolderName + "/HcalHaloData");
  if (StandardDQM) {
    hHcalHaloData_PhiWedgeMultiplicity = ibooker.book1D("HcalHaloData_PhiWedgeMultiplicity", "", 20, -0.5, 19.5);
    hHcalHaloData_PhiWedgeConstituents = ibooker.book1D("HcalHaloData_PhiWedgeConstituents", "", 20, -0.5, 19.5);
    //hHcalHaloData_PhiWedgeiPhi         = ibooker.book1D("HcalHaloData_PhiWedgeiPhi","", 72, 0.5,72.5);
    hHcalHaloData_PhiWedgeZDirectionConfidence =
        ibooker.book1D("HcalHaloData_ZDirectionConfidence", "", 120, -1.2, 1.2);
  } else {
    hHcalHaloData_PhiWedgeMultiplicity = ibooker.book1D("HcalHaloData_PhiWedgeMultiplicity", "", 20, -0.5, 19.5);
    hHcalHaloData_PhiWedgeEnergy = ibooker.book1D("HcalHaloData_PhiWedgeEnergy", "", 50, -0.5, 199.5);
    hHcalHaloData_PhiWedgeConstituents = ibooker.book1D("HcalHaloData_PhiWedgeConstituents", "", 20, -0.5, 19.5);
    hHcalHaloData_PhiWedgeiPhi = ibooker.book1D("HcalHaloData_PhiWedgeiPhi", "", 72, 0.5, 72.5);
    hHcalHaloData_PhiWedgeMinTime = ibooker.book1D("HcalHaloData_PhiWedgeMinTime", "", 50, -100.0, 100.0);
    hHcalHaloData_PhiWedgeMaxTime = ibooker.book1D("HcalHaloData_PhiWedgeMaxTime", "", 50, -100.0, 100.0);
    hHcalHaloData_PhiWedgePlusZDirectionConfidence =
        ibooker.book1D("HcalHaloData_PlusZDirectionConfidence", "", 50, 0., 1.0);
    hHcalHaloData_PhiWedgeZDirectionConfidence =
        ibooker.book1D("HcalHaloData_ZDirectionConfidence", "", 120, -1.2, 1.2);
    hHcalHaloData_PhiWedgeMinVsMaxTime =
        ibooker.book2D("HcalHaloData_PhiWedgeMinVsMaxTime", "", 50, -100.0, 100.0, 50, -100.0, 100.0);
  }

  // CSCHaloData
  ibooker.setCurrentFolder(FolderName + "/CSCHaloData");
  if (StandardDQM) {
    hCSCHaloData_TrackMultiplicity = ibooker.book1D("CSCHaloData_TrackMultiplicity", "", 15, -0.5, 14.5);
    hCSCHaloData_TrackMultiplicityMEPlus = ibooker.book1D("CSCHaloData_TrackMultiplicityMEPlus", "", 15, -0.5, 14.5);
    hCSCHaloData_TrackMultiplicityMEMinus = ibooker.book1D("CSCHaloData_TrackMultiplicityMEMinus", "", 15, -0.5, 14.5);
    hCSCHaloData_InnerMostTrackHitR = ibooker.book1D("CSCHaloData_InnerMostTrackHitR", "", 70, 99.5, 799.5);
    hCSCHaloData_InnerMostTrackHitPhi = ibooker.book1D("CSCHaloData_InnerMostTrackHitPhi", "", 60, -3.2, 3.2);
    hCSCHaloData_L1HaloTriggersMEPlus = ibooker.book1D("CSCHaloData_L1HaloTriggersMEPlus", "", 10, -0.5, 9.5);
    hCSCHaloData_L1HaloTriggersMEMinus = ibooker.book1D("CSCHaloData_L1HaloTriggersMEMinus", "", 10, -0.5, 9.5);
    hCSCHaloData_L1HaloTriggers = ibooker.book1D("CSCHaloData_L1HaloTriggers", "", 10, -0.5, 9.5);
    hCSCHaloData_HLHaloTriggers = ibooker.book1D("CSCHaloData_HLHaloTriggers", "", 2, -0.5, 1.5);
    hCSCHaloData_NOutOfTimeTriggersvsL1HaloExists =
        ibooker.book2D("CSCHaloData_NOutOfTimeTriggersvsL1HaloExists", "", 20, -0.5, 19.5, 2, -0.5, 1.5);
    hCSCHaloData_NOutOfTimeTriggersMEPlus = ibooker.book1D("CSCHaloData_NOutOfTimeTriggersMEPlus", "", 20, -0.5, 19.5);
    hCSCHaloData_NOutOfTimeTriggersMEMinus =
        ibooker.book1D("CSCHaloData_NOutOfTimeTriggersMEMinus", "", 20, -0.5, 19.5);
    hCSCHaloData_NOutOfTimeTriggers = ibooker.book1D("CSCHaloData_NOutOfTimeTriggers", "", 20, -0.5, 19.5);
    hCSCHaloData_NOutOfTimeHits = ibooker.book1D("CSCHaloData_NOutOfTimeHits", "", 60, -0.5, 59.5);
    hCSCHaloData_NTracksSmalldT = ibooker.book1D("CSCHaloData_NTracksSmalldT", "", 15, -0.5, 14.5);
    hCSCHaloData_NTracksSmallBeta = ibooker.book1D("CSCHaloData_NTracksSmallBeta", "", 15, -0.5, 14.5);
    hCSCHaloData_NTracksSmallBetaAndSmalldT =
        ibooker.book1D("CSCHaloData_NTracksSmallBetaAndSmalldT", "", 15, -0.5, 14.5);
    hCSCHaloData_NTracksSmalldTvsNHaloTracks =
        ibooker.book2D("CSCHaloData_NTracksSmalldTvsNHaloTracks", "", 15, -0.5, 14.5, 15, -0.5, 14.5);
    hCSCHaloData_SegmentdT = ibooker.book1D("CSCHaloData_SegmentdT", "", 100, -100, 100);
    hCSCHaloData_FreeInverseBeta = ibooker.book1D("CSCHaloData_FreeInverseBeta", "", 80, -4, 4);
    hCSCHaloData_FreeInverseBetaVsSegmentdT =
        ibooker.book2D("CSCHaloData_FreeInverseBetaVsSegmentdT", "", 100, -100, 100, 80, -4, 4);
    // MLR
    hCSCHaloData_NFlatHaloSegments = ibooker.book1D("CSCHaloData_NFlatHaloSegments", "", 20, 0, 20);
    hCSCHaloData_SegmentsInBothEndcaps = ibooker.book1D("CSCHaloData_SegmentsInBothEndcaps", "", 2, 0, 2);
    hCSCHaloData_NFlatSegmentsInBothEndcaps = ibooker.book1D("CSCHaloData_NFlatSegmentsInBothEndcaps", "", 20, 0, 20);
    // End MLR
  } else {
    hCSCHaloData_TrackMultiplicity = ibooker.book1D("CSCHaloData_TrackMultiplicity", "", 15, -0.5, 14.5);
    hCSCHaloData_TrackMultiplicityMEPlus = ibooker.book1D("CSCHaloData_TrackMultiplicityMEPlus", "", 15, -0.5, 14.5);
    hCSCHaloData_TrackMultiplicityMEMinus = ibooker.book1D("CSCHaloData_TrackMultiplicityMEMinus", "", 15, -0.5, 14.5);
    hCSCHaloData_InnerMostTrackHitXY =
        ibooker.book2D("CSCHaloData_InnerMostTrackHitXY", "", 100, -700, 700, 100, -700, 700);
    hCSCHaloData_InnerMostTrackHitR = ibooker.book1D("CSCHaloData_InnerMostTrackHitR", "", 400, -0.5, 799.5);
    hCSCHaloData_InnerMostTrackHitRPlusZ =
        ibooker.book2D("CSCHaloData_InnerMostTrackHitRPlusZ", "", 400, 400, 1200, 400, -0.5, 799.5);
    hCSCHaloData_InnerMostTrackHitRMinusZ =
        ibooker.book2D("CSCHaloData_InnerMostTrackHitRMinusZ", "", 400, -1200, -400, 400, -0.5, 799.5);
    hCSCHaloData_InnerMostTrackHitiPhi = ibooker.book1D("CSCHaloData_InnerMostTrackHitiPhi", "", 72, 0.5, 72.5);
    hCSCHaloData_InnerMostTrackHitPhi = ibooker.book1D("CSCHaloData_InnerMostTrackHitPhi", "", 60, -3.2, 3.2);
    hCSCHaloData_L1HaloTriggersMEPlus = ibooker.book1D("CSCHaloData_L1HaloTriggersMEPlus", "", 10, -0.5, 9.5);
    hCSCHaloData_L1HaloTriggersMEMinus = ibooker.book1D("CSCHaloData_L1HaloTriggersMEMinus", "", 10, -0.5, 9.5);
    hCSCHaloData_L1HaloTriggers = ibooker.book1D("CSCHaloData_L1HaloTriggers", "", 10, -0.5, 9.5);
    hCSCHaloData_HLHaloTriggers = ibooker.book1D("CSCHaloData_HLHaloTriggers", "", 2, -0.5, 1.5);
    hCSCHaloData_NOutOfTimeTriggersvsL1HaloExists =
        ibooker.book2D("CSCHaloData_NOutOfTimeTriggersvsL1HaloExists", "", 20, -0.5, 19.5, 2, -0.5, 1.5);
    hCSCHaloData_NOutOfTimeTriggers = ibooker.book1D("CSCHaloData_NOutOfTimeTriggers", "", 20, -0.5, 19.5);
    hCSCHaloData_NOutOfTimeHits = ibooker.book1D("CSCHaloData_NOutOfTimeHits", "", 60, -0.5, 59.5);
    hCSCHaloData_NTracksSmalldT = ibooker.book1D("CSCHaloData_NTracksSmalldT", "", 15, -0.5, 14.5);
    hCSCHaloData_NTracksSmallBeta = ibooker.book1D("CSCHaloData_NTracksSmallBeta", "", 15, -0.5, 14.5);
    hCSCHaloData_NTracksSmallBetaAndSmalldT =
        ibooker.book1D("CSCHaloData_NTracksSmallBetaAndSmalldT", "", 15, -0.5, 14.5);
    hCSCHaloData_NTracksSmalldTvsNHaloTracks =
        ibooker.book2D("CSCHaloData_NTracksSmalldTvsNHaloTracks", "", 15, -0.5, 14.5, 15, -0.5, 14.5);
    hCSCHaloData_SegmentdT = ibooker.book1D("CSCHaloData_SegmentdT", "", 100, -100, 100);
    hCSCHaloData_FreeInverseBeta = ibooker.book1D("CSCHaloData_FreeInverseBeta", "", 80, -4, 4);
    hCSCHaloData_FreeInverseBetaVsSegmentdT =
        ibooker.book2D("CSCHaloData_FreeInverseBetaVsSegmentdT", "", 100, -100, 100, 80, -4, 4);
    // MLR
    hCSCHaloData_NFlatHaloSegments = ibooker.book1D("CSCHaloData_NFlatHaloSegments", "", 20, 0, 20);
    hCSCHaloData_SegmentsInBothEndcaps = ibooker.book1D("CSCHaloData_SegmentsInBothEndcaps", "", 2, 0, 2);
    hCSCHaloData_NFlatSegmentsInBothEndcaps = ibooker.book1D("CSCHaloData_NFlatSegmentsInBothEndcaps", "", 20, 0, 20);
    // End MLR
  }

  // GlobalHaloData
  ibooker.setCurrentFolder(FolderName + "/GlobalHaloData");
  if (!StandardDQM) {
    hGlobalHaloData_MExCorrection = ibooker.book1D("GlobalHaloData_MExCorrection", "", 200, -200., 200.);
    hGlobalHaloData_MEyCorrection = ibooker.book1D("GlobalHaloData_MEyCorrection", "", 200, -200., 200.);
    hGlobalHaloData_SumEtCorrection = ibooker.book1D("GlobalHaloData_SumEtCorrection", "", 200, -0.5, 399.5);
    hGlobalHaloData_HaloCorrectedMET = ibooker.book1D("GlobalHaloData_HaloCorrectedMET", "", 500, -0.5, 1999.5);
    hGlobalHaloData_RawMETMinusHaloCorrectedMET =
        ibooker.book1D("GlobalHaloData_RawMETMinusHaloCorrectedMET", "", 250, -500., 500.);
    hGlobalHaloData_RawMETOverSumEt = ibooker.book1D("GlobalHaloData_RawMETOverSumEt", "", 100, 0.0, 1.0);
    hGlobalHaloData_MatchedHcalPhiWedgeMultiplicity =
        ibooker.book1D("GlobalHaloData_MatchedHcalPhiWedgeMultiplicity", "", 15, -0.5, 14.5);
    hGlobalHaloData_MatchedHcalPhiWedgeEnergy =
        ibooker.book1D("GlobalHaloData_MatchedHcalPhiWedgeEnergy", "", 50, -0.5, 199.5);
    hGlobalHaloData_MatchedHcalPhiWedgeConstituents =
        ibooker.book1D("GlobalHaloData_MatchedHcalPhiWedgeConstituents", "", 20, -0.5, 19.5);
    hGlobalHaloData_MatchedHcalPhiWedgeiPhi =
        ibooker.book1D("GlobalHaloData_MatchedHcalPhiWedgeiPhi", "", 1, 0.5, 72.5);
    hGlobalHaloData_MatchedHcalPhiWedgeMinTime =
        ibooker.book1D("GlobalHaloData_MatchedHcalPhiWedgeMinTime", "", 50, -100.0, 100.0);
    hGlobalHaloData_MatchedHcalPhiWedgeMaxTime =
        ibooker.book1D("GlobalHaloData_MatchedHcalPhiWedgeMaxTime", "", 50, -100.0, 100.0);
    hGlobalHaloData_MatchedHcalPhiWedgeZDirectionConfidence =
        ibooker.book1D("GlobalHaloData_MatchedHcalPhiWedgeZDirectionConfidence", "", 120, -1.2, 1.2);
    hGlobalHaloData_MatchedEcalPhiWedgeMultiplicity =
        ibooker.book1D("GlobalHaloData_MatchedEcalPhiWedgeMultiplicity", "", 15, -0.5, 14.5);
    hGlobalHaloData_MatchedEcalPhiWedgeEnergy =
        ibooker.book1D("GlobalHaloData_MatchedEcalPhiWedgeEnergy", "", 50, -0.5, 199.5);
    hGlobalHaloData_MatchedEcalPhiWedgeConstituents =
        ibooker.book1D("GlobalHaloData_MatchedEcalPhiWedgeConstituents", "", 20, -0.5, 19.5);
    hGlobalHaloData_MatchedEcalPhiWedgeiPhi =
        ibooker.book1D("GlobalHaloData_MatchedEcalPhiWedgeiPhi", "", 360, 0.5, 360.5);
    hGlobalHaloData_MatchedEcalPhiWedgeMinTime =
        ibooker.book1D("GlobalHaloData_MatchedEcalPhiWedgeMinTime", "", 50, -100.0, 100.0);
    hGlobalHaloData_MatchedEcalPhiWedgeMaxTime =
        ibooker.book1D("GlobalHaloData_MatchedEcalPhiWedgeMaxTime", "", 50, -100.0, 100.0);
    hGlobalHaloData_MatchedEcalPhiWedgeZDirectionConfidence =
        ibooker.book1D("GlobalHaloData_MatchedEcalPhiWedgeZDirectionConfidence", "", 120, 1.2, 1.2);
  }
  // BeamHaloSummary
  ibooker.setCurrentFolder(FolderName + "/BeamHaloSummary");

  hBeamHaloSummary_Id = ibooker.book1D("BeamHaloSumamry_Id", "", 11, 0.5, 11.5);
  hBeamHaloSummary_Id->setBinLabel(1, "CSC Loose");
  hBeamHaloSummary_Id->setBinLabel(2, "CSC Tight");
  hBeamHaloSummary_Id->setBinLabel(3, "Ecal Loose");
  hBeamHaloSummary_Id->setBinLabel(4, "Ecal Tight");
  hBeamHaloSummary_Id->setBinLabel(5, "Hcal Loose");
  hBeamHaloSummary_Id->setBinLabel(6, "Hcal Tight");
  hBeamHaloSummary_Id->setBinLabel(7, "Global Loose");
  hBeamHaloSummary_Id->setBinLabel(8, "Global Tight");
  hBeamHaloSummary_Id->setBinLabel(9, "Event Loose");
  hBeamHaloSummary_Id->setBinLabel(10, "Event Tight");
  hBeamHaloSummary_Id->setBinLabel(11, "Nothing");
  if (!StandardDQM) {
    hBeamHaloSummary_BXN = ibooker.book2D("BeamHaloSummary_BXN", "", 11, 0.5, 11.5, 4000, -0.5, 3999.5);
    hBeamHaloSummary_BXN->setBinLabel(1, "CSC Loose");
    hBeamHaloSummary_BXN->setBinLabel(2, "CSC Tight");
    hBeamHaloSummary_BXN->setBinLabel(3, "Ecal Loose");
    hBeamHaloSummary_BXN->setBinLabel(4, "Ecal Tight");
    hBeamHaloSummary_BXN->setBinLabel(5, "Hcal Loose");
    hBeamHaloSummary_BXN->setBinLabel(6, "Hcal Tight");
    hBeamHaloSummary_BXN->setBinLabel(7, "Global Loose");
    hBeamHaloSummary_BXN->setBinLabel(8, "Global Tight");
    hBeamHaloSummary_BXN->setBinLabel(9, "Event Loose");
    hBeamHaloSummary_BXN->setBinLabel(10, "Event Tight");
    hBeamHaloSummary_BXN->setBinLabel(11, "Nothing");
  }
  // Extra
  ibooker.setCurrentFolder(FolderName + "/ExtraHaloData");
  if (StandardDQM) {
    hExtra_CSCTrackInnerOuterDPhi = ibooker.book1D("Extra_CSCTrackInnerOuterDPhi", "", 30, 0, 3.2);
    hExtra_CSCTrackInnerOuterDEta = ibooker.book1D("Extra_CSCTrackInnerOuterDEta", "", 100, 0, 3.0);
    hExtra_CSCTrackChi2Ndof = ibooker.book1D("Extra_CSCTrackChi2Ndof", "", 25, 0, 10);
    hExtra_CSCTrackNHits = ibooker.book1D("Extra_CSCTrackNHits", "", 75, 0, 75);
    hExtra_CSCActivityWithMET = ibooker.book2D("Extra_CSCActivityWithMET", "", 4, 0.5, 4.5, 4, 0.5, 4.5);
    hExtra_CSCActivityWithMET->setBinLabel(1, "Track", 1);
    hExtra_CSCActivityWithMET->setBinLabel(1, "Track", 2);
    hExtra_CSCActivityWithMET->setBinLabel(2, "Segments", 1);
    hExtra_CSCActivityWithMET->setBinLabel(2, "Segments", 2);
    hExtra_CSCActivityWithMET->setBinLabel(3, "RecHits", 1);
    hExtra_CSCActivityWithMET->setBinLabel(3, "RecHits", 2);
    hExtra_CSCActivityWithMET->setBinLabel(4, "Nothing", 1);
    hExtra_CSCActivityWithMET->setBinLabel(4, "Nothing", 2);
    hExtra_InnerMostTrackHitR = ibooker.book1D("Extra_InnerMostTrackHitR", "", 70, 99.5, 799.5);
    hExtra_InnerMostTrackHitPhi = ibooker.book1D("Extra_InnerMostTrackHitPhi", "", 60, -3.2, 3.2);
  } else {
    hExtra_CSCActivityWithMET = ibooker.book2D("Extra_CSCActivityWithMET", "", 4, 0.5, 4.5, 4, 0.5, 4.5);
    hExtra_CSCActivityWithMET->setBinLabel(1, "Track", 1);
    hExtra_CSCActivityWithMET->setBinLabel(1, "Track", 2);
    hExtra_CSCActivityWithMET->setBinLabel(2, "Segments", 1);
    hExtra_CSCActivityWithMET->setBinLabel(2, "Segments", 2);
    hExtra_CSCActivityWithMET->setBinLabel(3, "RecHits", 1);
    hExtra_CSCActivityWithMET->setBinLabel(3, "RecHits", 2);
    hExtra_CSCActivityWithMET->setBinLabel(4, "Nothing", 1);
    hExtra_CSCActivityWithMET->setBinLabel(4, "Nothing", 2);
    hExtra_HcalToF = ibooker.book2D("Extra_HcalToF", "", 83, -41.5, 41.5, 1000, -125., 125.);
    hExtra_HcalToF_HaloId = ibooker.book2D("Extra_HcalToF_HaloId", "", 83, -41.5, 41.5, 1000, -125., 125.);
    hExtra_EcalToF = ibooker.book2D("Extra_EcalToF", "", 171, -85.5, 85.5, 2000, -225., 225.);
    hExtra_EcalToF_HaloId = ibooker.book2D("Extra_EcalToF_HaloId", "", 171, -85.5, 85.5, 2000, -225., 225.);
    hExtra_CSCTrackInnerOuterDPhi = ibooker.book1D("Extra_CSCTrackInnerOuterDPhi", "", 30, 0, 3.2);
    hExtra_CSCTrackInnerOuterDEta = ibooker.book1D("Extra_CSCTrackInnerOuterDEta", "", 30, 0, 3.2);
    hExtra_CSCTrackChi2Ndof = ibooker.book1D("Extra_CSCTrackChi2Ndof", "", 100, 0, 10);
    hExtra_CSCTrackNHits = ibooker.book1D("Extra_CSCTrackNHits", "", 75, 0, 75);
    hExtra_InnerMostTrackHitXY = ibooker.book2D("Extra_InnerMostTrackHitXY", "", 100, -700, 700, 100, -700, 700);
    hExtra_InnerMostTrackHitR = ibooker.book1D("Extra_InnerMostTrackHitR", "", 400, -0.5, 799.5);
    hExtra_InnerMostTrackHitRPlusZ =
        ibooker.book2D("Extra_InnerMostTrackHitRPlusZ", "", 400, 400, 1200, 400, -0.5, 799.5);
    hExtra_InnerMostTrackHitRMinusZ =
        ibooker.book2D("Extra_InnerMostTrackHitRMinusZ", "", 400, -1200, -400, 400, -0.5, 799.5);
    hExtra_InnerMostTrackHitiPhi = ibooker.book1D("Extra_InnerMostTrackHitiPhi", "", 72, 0.5, 72.5);
    hExtra_InnerMostTrackHitPhi = ibooker.book1D("Extra_InnerMostTrackHitPhi", "", 60, -3.2, 3.2);
    hExtra_BXN = ibooker.book1D("Extra_BXN", "BXN Occupancy", 4000, 0.5, 4000.5);
  }
}

void BeamHaloAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  EventID TheEvent = iEvent.id();
  int BXN = iEvent.bunchCrossing();
  bool Dump = !TextFileName.empty();
  edm::EventNumber_t TheEventNumber = TheEvent.event();
  edm::LuminosityBlockNumber_t Lumi = iEvent.luminosityBlock();
  edm::RunNumber_t Run = iEvent.run();

  //Get CSC Geometry
  const auto& TheCSCGeometry = iSetup.getHandle(cscGeomToken_);
  //Note - removed getting calogeometry since it was unused
  //Get Stand-alone Muons from Cosmic Muon Reconstruction
  edm::Handle<reco::MuonCollection> TheCosmics;
  iEvent.getByToken(IT_CosmicStandAloneMuon, TheCosmics);
  edm::Handle<reco::MuonTimeExtraMap> TheCSCTimeMap;
  iEvent.getByToken(IT_CSCTimeMapToken, TheCSCTimeMap);
  bool CSCTrackPlus = false;
  bool CSCTrackMinus = false;
  int imucount = 0;
  if (TheCosmics.isValid()) {
    for (reco::MuonCollection::const_iterator iMuon = TheCosmics->begin(); iMuon != TheCosmics->end();
         iMuon++, imucount++) {
      reco::TrackRef Track = iMuon->outerTrack();
      if (!Track)
        continue;

      if (!CSCTrackPlus || !CSCTrackMinus) {
        if (Track->eta() > 0 || Track->outerPosition().z() > 0 || Track->innerPosition().z() > 0)
          CSCTrackPlus = true;
        else if (Track->eta() < 0 || Track->outerPosition().z() < 0 || Track->innerPosition().z() < 0)
          CSCTrackMinus = true;
      }

      float innermost_phi = 0.;
      float outermost_phi = 0.;
      float innermost_z = 99999.;
      float outermost_z = 0.;
      float innermost_eta = 0.;
      float outermost_eta = 0.;
      float innermost_x = 0.;
      float innermost_y = 0.;
      float innermost_r = 0.;
      for (unsigned int j = 0; j < Track->extra()->recHitsSize(); j++) {
        auto hit = Track->extra()->recHitRef(j);
        DetId TheDetUnitId(hit->geographicalId());
        if (TheDetUnitId.det() != DetId::Muon)
          continue;
        if (TheDetUnitId.subdetId() != MuonSubdetId::CSC)
          continue;

        const GeomDetUnit* TheUnit = TheCSCGeometry->idToDetUnit(TheDetUnitId);
        LocalPoint TheLocalPosition = hit->localPosition();
        const BoundPlane& TheSurface = TheUnit->surface();
        const GlobalPoint TheGlobalPosition = TheSurface.toGlobal(TheLocalPosition);

        float z = TheGlobalPosition.z();
        if (TMath::Abs(z) < innermost_z) {
          innermost_phi = TheGlobalPosition.phi();
          innermost_eta = TheGlobalPosition.eta();
          innermost_z = TheGlobalPosition.z();
          innermost_x = TheGlobalPosition.x();
          innermost_y = TheGlobalPosition.y();
          innermost_r = TMath::Sqrt(innermost_x * innermost_x + innermost_y * innermost_y);
        }
        if (TMath::Abs(z) > outermost_z) {
          outermost_phi = TheGlobalPosition.phi();
          outermost_eta = TheGlobalPosition.eta();
          outermost_z = TheGlobalPosition.z();
        }
      }
      float dphi = TMath::Abs(outermost_phi - innermost_phi);
      float deta = TMath::Abs(outermost_eta - innermost_eta);
      hExtra_CSCTrackInnerOuterDPhi->Fill(dphi);
      hExtra_CSCTrackInnerOuterDEta->Fill(deta);
      hExtra_CSCTrackChi2Ndof->Fill(Track->normalizedChi2());
      hExtra_CSCTrackNHits->Fill(Track->numberOfValidHits());
      hExtra_InnerMostTrackHitR->Fill(innermost_r);
      hExtra_InnerMostTrackHitPhi->Fill(innermost_phi);
      if (!StandardDQM) {
        hExtra_InnerMostTrackHitXY->Fill(innermost_x, innermost_y);
        hExtra_InnerMostTrackHitiPhi->Fill(Phi_To_iPhi(innermost_phi));
        if (innermost_z > 0)
          hExtra_InnerMostTrackHitRPlusZ->Fill(innermost_z, innermost_r);
        else
          hExtra_InnerMostTrackHitRMinusZ->Fill(innermost_z, innermost_r);
      }

      std::vector<const CSCSegment*> MatchedSegments = TheMatcher->matchCSC(*Track, iEvent);
      // Find the inner and outer segments separately in case they don't agree completely with recHits
      // Plan for the possibility segments in both endcaps
      float InnerSegmentTime[2] = {0, 0};
      float OuterSegmentTime[2] = {0, 0};
      float innermost_seg_z[2] = {1500, 1500};
      float outermost_seg_z[2] = {0, 0};
      for (std::vector<const CSCSegment*>::const_iterator segment = MatchedSegments.begin();
           segment != MatchedSegments.end();
           ++segment) {
        CSCDetId TheCSCDetId((*segment)->cscDetId());
        const CSCChamber* TheCSCChamber = TheCSCGeometry->chamber(TheCSCDetId);
        LocalPoint TheLocalPosition = (*segment)->localPosition();
        const GlobalPoint TheGlobalPosition = TheCSCChamber->toGlobal(TheLocalPosition);
        float z = TheGlobalPosition.z();
        int TheEndcap = TheCSCDetId.endcap();
        if (TMath::Abs(z) < innermost_seg_z[TheEndcap - 1]) {
          innermost_seg_z[TheEndcap - 1] = TMath::Abs(z);
          InnerSegmentTime[TheEndcap - 1] = (*segment)->time();
        }
        if (TMath::Abs(z) > outermost_seg_z[TheEndcap - 1]) {
          outermost_seg_z[TheEndcap - 1] = TMath::Abs(z);
          OuterSegmentTime[TheEndcap - 1] = (*segment)->time();
        }
      }

      float dT_Segment = 0;                         // default safe value, looks like collision muon
      if (innermost_seg_z[0] < outermost_seg_z[0])  // two segments in ME+
        dT_Segment = OuterSegmentTime[0] - InnerSegmentTime[0];
      if (innermost_seg_z[1] < outermost_seg_z[1])  // two segments in ME-
      {
        // replace the measurement if there weren't segments in ME+ or
        // if the track in ME- has timing more consistent with an incoming particle
        if (dT_Segment == 0.0 || OuterSegmentTime[1] - InnerSegmentTime[1] < dT_Segment)
          dT_Segment = OuterSegmentTime[1] - InnerSegmentTime[1];
      }
      hCSCHaloData_SegmentdT->Fill(dT_Segment);

      // Analyze the MuonTimeExtra information
      reco::MuonRef muonR(TheCosmics, imucount);
      if (TheCSCTimeMap.isValid()) {
        const reco::MuonTimeExtraMap& timeMapCSC = *TheCSCTimeMap;
        reco::MuonTimeExtra timecsc = timeMapCSC[muonR];
        float freeInverseBeta = timecsc.freeInverseBeta();
        hCSCHaloData_FreeInverseBeta->Fill(freeInverseBeta);
        hCSCHaloData_FreeInverseBetaVsSegmentdT->Fill(dT_Segment, freeInverseBeta);
      }
    }
  }

  //Get CSC Segments
  edm::Handle<CSCSegmentCollection> TheCSCSegments;
  iEvent.getByToken(IT_CSCSegment, TheCSCSegments);

  // Group segments according to endcaps
  std::vector<CSCSegment> vCSCSegments_Plus;
  std::vector<CSCSegment> vCSCSegments_Minus;

  bool CSCSegmentPlus = false;
  bool CSCSegmentMinus = false;
  if (TheCSCSegments.isValid()) {
    for (CSCSegmentCollection::const_iterator iSegment = TheCSCSegments->begin(); iSegment != TheCSCSegments->end();
         iSegment++) {
      const std::vector<CSCRecHit2D> vCSCRecHits = iSegment->specificRecHits();
      CSCDetId iDetId = (CSCDetId)(*iSegment).cscDetId();

      if (iDetId.endcap() == 1)
        vCSCSegments_Plus.push_back(*iSegment);
      else
        vCSCSegments_Minus.push_back(*iSegment);
    }
  }

  // Are there segments on the plus/minus side?
  if (!vCSCSegments_Plus.empty())
    CSCSegmentPlus = true;
  if (!vCSCSegments_Minus.empty())
    CSCSegmentMinus = true;

  //Get CSC RecHits
  Handle<CSCRecHit2DCollection> TheCSCRecHits;
  iEvent.getByToken(IT_CSCRecHit, TheCSCRecHits);
  bool CSCRecHitPlus = false;
  bool CSCRecHitMinus = false;
  if (TheCSCRecHits.isValid()) {
    for (CSCRecHit2DCollection::const_iterator iCSCRecHit = TheCSCRecHits->begin(); iCSCRecHit != TheCSCRecHits->end();
         iCSCRecHit++) {
      DetId TheDetUnitId(iCSCRecHit->geographicalId());
      const GeomDetUnit* TheUnit = (*TheCSCGeometry).idToDetUnit(TheDetUnitId);
      LocalPoint TheLocalPosition = iCSCRecHit->localPosition();
      const BoundPlane& TheSurface = TheUnit->surface();
      GlobalPoint TheGlobalPosition = TheSurface.toGlobal(TheLocalPosition);

      //Are there hits on the plus/minus side?
      if (TheGlobalPosition.z() > 0)
        CSCRecHitPlus = true;
      else
        CSCRecHitMinus = true;
    }
  }

  //Get  EB RecHits
  edm::Handle<EBRecHitCollection> TheEBRecHits;
  iEvent.getByToken(IT_EBRecHit, TheEBRecHits);
  int EBHits = 0;
  if (TheEBRecHits.isValid()) {
    for (EBRecHitCollection::const_iterator iEBRecHit = TheEBRecHits->begin(); iEBRecHit != TheEBRecHits->end();
         iEBRecHit++) {
      if (iEBRecHit->energy() < 0.5)
        continue;
      DetId id = DetId(iEBRecHit->id());
      EBDetId EcalId(id.rawId());
      int ieta = EcalId.ieta();
      if (!StandardDQM)
        hExtra_EcalToF->Fill(ieta, iEBRecHit->time());
      EBHits++;
    }
  }

  //Get HB/HE RecHits
  edm::Handle<HBHERecHitCollection> TheHBHERecHits;
  iEvent.getByToken(IT_HBHERecHit, TheHBHERecHits);
  if (TheHBHERecHits.isValid()) {
    for (HBHERecHitCollection::const_iterator iHBHERecHit = TheHBHERecHits->begin();
         iHBHERecHit != TheHBHERecHits->end();
         iHBHERecHit++) {
      if (iHBHERecHit->energy() < 1.)
        continue;
      HcalDetId id = HcalDetId(iHBHERecHit->id());
      if (!StandardDQM)
        hExtra_HcalToF->Fill(id.ieta(), iHBHERecHit->time());
    }
  }

  //Get MET
  edm::Handle<reco::CaloMETCollection> TheCaloMET;
  iEvent.getByToken(IT_met, TheCaloMET);

  //Get CSCHaloData
  edm::Handle<reco::CSCHaloData> TheCSCDataHandle;
  iEvent.getByToken(IT_CSCHaloData, TheCSCDataHandle);
  int TheHaloOrigin = 0;
  if (TheCSCDataHandle.isValid()) {
    const CSCHaloData CSCData = (*TheCSCDataHandle.product());
    if (CSCData.NumberOfOutOfTimeTriggers(HaloData::plus) && !CSCData.NumberOfOutOfTimeTriggers(HaloData::minus))
      TheHaloOrigin = 1;
    else if (CSCData.NumberOfOutOfTimeTriggers(HaloData::minus) && !CSCData.NumberOfOutOfTimeTriggers(HaloData::plus))
      TheHaloOrigin = -1;

    for (std::vector<GlobalPoint>::const_iterator i = CSCData.GetCSCTrackImpactPositions().begin();
         i != CSCData.GetCSCTrackImpactPositions().end();
         i++) {
      float r = TMath::Sqrt(i->x() * i->x() + i->y() * i->y());
      if (!StandardDQM) {
        hCSCHaloData_InnerMostTrackHitXY->Fill(i->x(), i->y());
        hCSCHaloData_InnerMostTrackHitiPhi->Fill(Phi_To_iPhi(i->phi()));
        if (i->z() > 0)
          hCSCHaloData_InnerMostTrackHitRPlusZ->Fill(i->z(), r);
        else
          hCSCHaloData_InnerMostTrackHitRMinusZ->Fill(i->z(), r);
      }
      hCSCHaloData_InnerMostTrackHitR->Fill(r);
      hCSCHaloData_InnerMostTrackHitPhi->Fill(i->phi());
    }
    hCSCHaloData_L1HaloTriggersMEPlus->Fill(CSCData.NumberOfHaloTriggers(HaloData::plus));
    hCSCHaloData_L1HaloTriggersMEMinus->Fill(CSCData.NumberOfHaloTriggers(HaloData::minus));
    hCSCHaloData_L1HaloTriggers->Fill(CSCData.NumberOfHaloTriggers(HaloData::both));
    hCSCHaloData_HLHaloTriggers->Fill(CSCData.CSCHaloHLTAccept());
    hCSCHaloData_TrackMultiplicityMEPlus->Fill(CSCData.NumberOfHaloTracks(HaloData::plus));
    hCSCHaloData_TrackMultiplicityMEMinus->Fill(CSCData.NumberOfHaloTracks(HaloData::minus));
    hCSCHaloData_TrackMultiplicity->Fill(CSCData.GetTracks().size());
    hCSCHaloData_NOutOfTimeTriggersMEPlus->Fill(CSCData.NOutOfTimeTriggers(HaloData::plus));
    hCSCHaloData_NOutOfTimeTriggersMEMinus->Fill(CSCData.NOutOfTimeTriggers(HaloData::minus));
    hCSCHaloData_NOutOfTimeTriggers->Fill(CSCData.NOutOfTimeTriggers(HaloData::both));
    hCSCHaloData_NOutOfTimeHits->Fill(CSCData.NOutOfTimeHits());
    hCSCHaloData_NOutOfTimeTriggersvsL1HaloExists->Fill(CSCData.NOutOfTimeTriggers(HaloData::both),
                                                        CSCData.NumberOfHaloTriggers(HaloData::both) > 0);
    hCSCHaloData_NTracksSmalldT->Fill(CSCData.NTracksSmalldT());
    hCSCHaloData_NTracksSmallBeta->Fill(CSCData.NTracksSmallBeta());
    hCSCHaloData_NTracksSmallBetaAndSmalldT->Fill(CSCData.NTracksSmallBetaAndSmalldT());
    hCSCHaloData_NTracksSmalldTvsNHaloTracks->Fill(CSCData.GetTracks().size(), CSCData.NTracksSmalldT());
    // MLR
    hCSCHaloData_NFlatHaloSegments->Fill(CSCData.NFlatHaloSegments());
    hCSCHaloData_SegmentsInBothEndcaps->Fill(CSCData.GetSegmentsInBothEndcaps());
    if (CSCData.GetSegmentsInBothEndcaps())
      hCSCHaloData_NFlatSegmentsInBothEndcaps->Fill(CSCData.NFlatHaloSegments());
    // End MLR
  }

  //Get EcalHaloData
  edm::Handle<reco::EcalHaloData> TheEcalHaloData;
  iEvent.getByToken(IT_EcalHaloData, TheEcalHaloData);
  if (TheEcalHaloData.isValid()) {
    const EcalHaloData EcalData = (*TheEcalHaloData.product());
    std::vector<PhiWedge> EcalWedges = EcalData.GetPhiWedges();
    for (std::vector<PhiWedge>::const_iterator iWedge = EcalWedges.begin(); iWedge != EcalWedges.end(); iWedge++) {
      if (!StandardDQM) {
        hEcalHaloData_PhiWedgeEnergy->Fill(iWedge->Energy());
        hEcalHaloData_PhiWedgeMinTime->Fill(iWedge->MinTime());
        hEcalHaloData_PhiWedgeMaxTime->Fill(iWedge->MaxTime());
        hEcalHaloData_PhiWedgeMinVsMaxTime->Fill(iWedge->MinTime(), iWedge->MaxTime());
        hEcalHaloData_PhiWedgePlusZDirectionConfidence->Fill(iWedge->PlusZDirectionConfidence());
        hEcalHaloData_PhiWedgeiPhi->Fill(iWedge->iPhi());
      }
      hEcalHaloData_PhiWedgeZDirectionConfidence->Fill(iWedge->ZDirectionConfidence());
      hEcalHaloData_PhiWedgeConstituents->Fill(iWedge->NumberOfConstituents());
    }

    hEcalHaloData_PhiWedgeMultiplicity->Fill(EcalWedges.size());

    edm::ValueMap<float> vm_Angle = EcalData.GetShowerShapesAngle();
    edm::ValueMap<float> vm_Roundness = EcalData.GetShowerShapesRoundness();
    //Access selected SuperClusters
    for (unsigned int n = 0; n < EcalData.GetSuperClusters().size(); n++) {
      edm::Ref<SuperClusterCollection> cluster(EcalData.GetSuperClusters()[n]);
      float angle = vm_Angle[cluster];
      float roundness = vm_Roundness[cluster];
      hEcalHaloData_SuperClusterShowerShapes->Fill(angle, roundness);
      hEcalHaloData_SuperClusterNHits->Fill(cluster->size());
      hEcalHaloData_SuperClusterEnergy->Fill(cluster->energy());

      if (!StandardDQM) {
        hEcalHaloData_SuperClusterPhiVsEta->Fill(cluster->eta(), cluster->phi());
      }
    }
  }

  //Get HcalHaloData
  edm::Handle<reco::HcalHaloData> TheHcalHaloData;
  iEvent.getByToken(IT_HcalHaloData, TheHcalHaloData);
  if (TheHcalHaloData.isValid()) {
    const HcalHaloData HcalData = (*TheHcalHaloData.product());
    std::vector<PhiWedge> HcalWedges = HcalData.GetPhiWedges();
    hHcalHaloData_PhiWedgeMultiplicity->Fill(HcalWedges.size());
    for (std::vector<PhiWedge>::const_iterator iWedge = HcalWedges.begin(); iWedge != HcalWedges.end(); iWedge++) {
      if (!StandardDQM) {
        hHcalHaloData_PhiWedgeEnergy->Fill(iWedge->Energy());
        hHcalHaloData_PhiWedgeMinTime->Fill(iWedge->MinTime());
        hHcalHaloData_PhiWedgeMaxTime->Fill(iWedge->MaxTime());
        hHcalHaloData_PhiWedgePlusZDirectionConfidence->Fill(iWedge->PlusZDirectionConfidence());
        hHcalHaloData_PhiWedgeMinVsMaxTime->Fill(iWedge->MinTime(), iWedge->MaxTime());
        hHcalHaloData_PhiWedgeiPhi->Fill(iWedge->iPhi());
      }

      hHcalHaloData_PhiWedgeConstituents->Fill(iWedge->NumberOfConstituents());
      hHcalHaloData_PhiWedgeZDirectionConfidence->Fill(iWedge->ZDirectionConfidence());
    }
  }

  if (!StandardDQM) {
    //Get GlobalHaloData
    edm::Handle<reco::GlobalHaloData> TheGlobalHaloData;
    iEvent.getByToken(IT_GlobalHaloData, TheGlobalHaloData);
    if (TheGlobalHaloData.isValid()) {
      const GlobalHaloData GlobalData = (*TheGlobalHaloData.product());
      if (TheCaloMET.isValid()) {
        // Get Raw Uncorrected CaloMET
        const CaloMETCollection* calometcol = TheCaloMET.product();
        const CaloMET* RawMET = &(calometcol->front());

        // Get BeamHalo Corrected CaloMET
        const CaloMET CorrectedMET = GlobalData.GetCorrectedCaloMET(*RawMET);
        hGlobalHaloData_MExCorrection->Fill(GlobalData.DeltaMEx());
        hGlobalHaloData_MEyCorrection->Fill(GlobalData.DeltaMEy());
        hGlobalHaloData_HaloCorrectedMET->Fill(CorrectedMET.pt());
        hGlobalHaloData_RawMETMinusHaloCorrectedMET->Fill(RawMET->pt() - CorrectedMET.pt());
        if (RawMET->sumEt())
          hGlobalHaloData_RawMETOverSumEt->Fill(RawMET->pt() / RawMET->sumEt());
      }

      // Get Matched Hcal Phi Wedges
      std::vector<PhiWedge> HcalWedges = GlobalData.GetMatchedHcalPhiWedges();
      hGlobalHaloData_MatchedHcalPhiWedgeMultiplicity->Fill(HcalWedges.size());
      // Loop over Matched Hcal Phi Wedges
      for (std::vector<PhiWedge>::const_iterator iWedge = HcalWedges.begin(); iWedge != HcalWedges.end(); iWedge++) {
        hGlobalHaloData_MatchedHcalPhiWedgeEnergy->Fill(iWedge->Energy());
        hGlobalHaloData_MatchedHcalPhiWedgeConstituents->Fill(iWedge->NumberOfConstituents());
        hGlobalHaloData_MatchedHcalPhiWedgeiPhi->Fill(iWedge->iPhi());
        hGlobalHaloData_MatchedHcalPhiWedgeMinTime->Fill(iWedge->MinTime());
        hGlobalHaloData_MatchedHcalPhiWedgeMaxTime->Fill(iWedge->MaxTime());
        hGlobalHaloData_MatchedHcalPhiWedgeZDirectionConfidence->Fill(iWedge->ZDirectionConfidence());
        if (TheHBHERecHits.isValid()) {
          for (HBHERecHitCollection::const_iterator iHBHERecHit = TheHBHERecHits->begin();
               iHBHERecHit != TheHBHERecHits->end();
               iHBHERecHit++) {
            HcalDetId id = HcalDetId(iHBHERecHit->id());
            int iphi = id.iphi();
            if (iphi != iWedge->iPhi())
              continue;
            if (iHBHERecHit->energy() < 1.0)
              continue;  // Otherwise there are thousands of hits per event (even with negative energies)

            float time = iHBHERecHit->time();
            int ieta = id.ieta();
            hExtra_HcalToF_HaloId->Fill(ieta, time);
          }
        }
      }

      // Get Matched Hcal Phi Wedges
      std::vector<PhiWedge> EcalWedges = GlobalData.GetMatchedEcalPhiWedges();
      hGlobalHaloData_MatchedEcalPhiWedgeMultiplicity->Fill(EcalWedges.size());
      for (std::vector<PhiWedge>::const_iterator iWedge = EcalWedges.begin(); iWedge != EcalWedges.end(); iWedge++) {
        hGlobalHaloData_MatchedEcalPhiWedgeEnergy->Fill(iWedge->Energy());
        hGlobalHaloData_MatchedEcalPhiWedgeConstituents->Fill(iWedge->NumberOfConstituents());
        hGlobalHaloData_MatchedEcalPhiWedgeiPhi->Fill(iWedge->iPhi());
        hGlobalHaloData_MatchedEcalPhiWedgeMinTime->Fill(iWedge->MinTime());
        hGlobalHaloData_MatchedEcalPhiWedgeMaxTime->Fill(iWedge->MaxTime());
        hGlobalHaloData_MatchedEcalPhiWedgeZDirectionConfidence->Fill(iWedge->ZDirectionConfidence());
        if (TheEBRecHits.isValid()) {
          for (EBRecHitCollection::const_iterator iEBRecHit = TheEBRecHits->begin(); iEBRecHit != TheEBRecHits->end();
               iEBRecHit++) {
            if (iEBRecHit->energy() < 0.5)
              continue;
            DetId id = DetId(iEBRecHit->id());
            EBDetId EcalId(id.rawId());
            int iPhi = EcalId.iphi();
            iPhi = (iPhi - 1) / 5 + 1;
            if (iPhi != iWedge->iPhi())
              continue;
            hExtra_EcalToF_HaloId->Fill(EcalId.ieta(), iEBRecHit->time());
          }
        }
      }
    }
  }

  // Get BeamHaloSummary
  edm::Handle<BeamHaloSummary> TheBeamHaloSummary;
  iEvent.getByToken(IT_BeamHaloSummary, TheBeamHaloSummary);
  if (TheBeamHaloSummary.isValid()) {
    const BeamHaloSummary TheSummary = (*TheBeamHaloSummary.product());
    if (TheSummary.CSCLooseHaloId()) {
      hBeamHaloSummary_Id->Fill(1);
      if (!StandardDQM)
        hBeamHaloSummary_BXN->Fill(1, BXN);
      if (Dump)
        *out << std::setw(15) << "CSCLoose" << std::setw(15) << Run << std::setw(15) << Lumi << std::setw(15)
             << TheEventNumber << std::endl;
    }
    if (TheSummary.CSCTightHaloId()) {
      hBeamHaloSummary_Id->Fill(2);
      if (!StandardDQM)
        hBeamHaloSummary_BXN->Fill(2, BXN);
    }
    if (TheSummary.EcalLooseHaloId()) {
      hBeamHaloSummary_Id->Fill(3);
      if (!StandardDQM)
        hBeamHaloSummary_BXN->Fill(3, BXN);
      if (Dump)
        *out << std::setw(15) << "EcalLoose" << std::setw(15) << Run << std::setw(15) << Lumi << std::setw(15)
             << TheEventNumber << std::endl;
    }
    if (TheSummary.EcalTightHaloId()) {
      hBeamHaloSummary_Id->Fill(4);
      if (!StandardDQM)
        hBeamHaloSummary_BXN->Fill(4, BXN);
    }
    if (TheSummary.HcalLooseHaloId()) {
      hBeamHaloSummary_Id->Fill(5);
      if (!StandardDQM)
        hBeamHaloSummary_BXN->Fill(5, BXN);
      if (Dump)
        *out << std::setw(15) << "HcalLoose" << std::setw(15) << Run << std::setw(15) << Lumi << std::setw(15)
             << TheEventNumber << std::endl;
    }
    if (TheSummary.HcalTightHaloId()) {
      hBeamHaloSummary_Id->Fill(6);
      if (!StandardDQM)
        hBeamHaloSummary_BXN->Fill(6, BXN);
    }
    if (TheSummary.GlobalLooseHaloId()) {
      hBeamHaloSummary_Id->Fill(7);
      if (!StandardDQM)
        hBeamHaloSummary_BXN->Fill(7, BXN);
      if (Dump)
        *out << std::setw(15) << "GlobalLoose" << std::setw(15) << Run << std::setw(15) << Lumi << std::setw(15)
             << TheEventNumber << std::endl;
    }
    if (TheSummary.GlobalTightHaloId()) {
      hBeamHaloSummary_Id->Fill(8);
      if (!StandardDQM)
        hBeamHaloSummary_BXN->Fill(8, BXN);
    }
    if (TheSummary.LooseId()) {
      hBeamHaloSummary_Id->Fill(9);
      if (!StandardDQM)
        hBeamHaloSummary_BXN->Fill(9, BXN);
    }
    if (TheSummary.TightId()) {
      hBeamHaloSummary_Id->Fill(10);
      if (!StandardDQM)
        hBeamHaloSummary_BXN->Fill(10, BXN);
    }
    if (!TheSummary.EcalLooseHaloId() && !TheSummary.HcalLooseHaloId() && !TheSummary.CSCLooseHaloId() &&
        !TheSummary.GlobalLooseHaloId()) {
      hBeamHaloSummary_Id->Fill(11);
      if (!StandardDQM)
        hBeamHaloSummary_BXN->Fill(11, BXN);
    }
  }

  if (TheCaloMET.isValid()) {
    const CaloMETCollection* calometcol = TheCaloMET.product();
    const CaloMET* calomet = &(calometcol->front());

    if (calomet->pt() > DumpMET)
      if (Dump)
        *out << std::setw(15) << "HighMET" << std::setw(15) << Run << std::setw(15) << Lumi << std::setw(15)
             << TheEventNumber << std::endl;

    //Fill CSC Activity Plot
    if (calomet->pt() > 15.0) {
      if (TheHaloOrigin > 0) {
        if (CSCTrackPlus && CSCTrackMinus)
          hExtra_CSCActivityWithMET->Fill(1, 1);
        else if (CSCTrackPlus && CSCSegmentMinus)
          hExtra_CSCActivityWithMET->Fill(1, 2);
        else if (CSCTrackPlus && CSCRecHitMinus)
          hExtra_CSCActivityWithMET->Fill(1, 3);
        else if (CSCTrackPlus)
          hExtra_CSCActivityWithMET->Fill(1, 4);
        else if (CSCSegmentPlus && CSCTrackMinus)
          hExtra_CSCActivityWithMET->Fill(2, 1);
        else if (CSCSegmentPlus && CSCSegmentMinus)
          hExtra_CSCActivityWithMET->Fill(2, 2);
        else if (CSCSegmentPlus && CSCRecHitMinus)
          hExtra_CSCActivityWithMET->Fill(2, 3);
        else if (CSCSegmentPlus)
          hExtra_CSCActivityWithMET->Fill(2, 4);
        else if (CSCRecHitPlus && CSCTrackMinus)
          hExtra_CSCActivityWithMET->Fill(3, 1);
        else if (CSCRecHitPlus && CSCSegmentMinus)
          hExtra_CSCActivityWithMET->Fill(3, 2);
        else if (CSCRecHitPlus && CSCRecHitMinus)
          hExtra_CSCActivityWithMET->Fill(3, 3);
        else if (CSCRecHitPlus)
          hExtra_CSCActivityWithMET->Fill(3, 4);
        else
          hExtra_CSCActivityWithMET->Fill(4, 4);
      } else if (TheHaloOrigin < 0) {
        if (CSCTrackMinus && CSCTrackPlus)
          hExtra_CSCActivityWithMET->Fill(1, 1);
        else if (CSCTrackMinus && CSCSegmentPlus)
          hExtra_CSCActivityWithMET->Fill(1, 2);
        else if (CSCTrackMinus && CSCRecHitPlus)
          hExtra_CSCActivityWithMET->Fill(1, 3);
        else if (CSCTrackMinus)
          hExtra_CSCActivityWithMET->Fill(1, 4);
        else if (CSCSegmentMinus && CSCTrackPlus)
          hExtra_CSCActivityWithMET->Fill(2, 1);
        else if (CSCSegmentMinus && CSCSegmentPlus)
          hExtra_CSCActivityWithMET->Fill(2, 2);
        else if (CSCSegmentMinus && CSCRecHitPlus)
          hExtra_CSCActivityWithMET->Fill(2, 3);
        else if (CSCSegmentMinus)
          hExtra_CSCActivityWithMET->Fill(2, 4);
        else if (CSCRecHitMinus && CSCTrackPlus)
          hExtra_CSCActivityWithMET->Fill(3, 1);
        else if (CSCRecHitMinus && CSCSegmentPlus)
          hExtra_CSCActivityWithMET->Fill(3, 2);
        else if (CSCRecHitMinus && CSCRecHitPlus)
          hExtra_CSCActivityWithMET->Fill(3, 3);
        else if (CSCRecHitMinus)
          hExtra_CSCActivityWithMET->Fill(3, 4);
        else
          hExtra_CSCActivityWithMET->Fill(4, 4);
      }
    }
  }
}

BeamHaloAnalyzer::~BeamHaloAnalyzer() {}

//DEFINE_FWK_MODULE(CMSEventAnalyzer);
