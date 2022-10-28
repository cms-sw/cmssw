#ifndef DQMOFFLINE_JETMET_BEAMHALO_ANALYZER_H
#define DQMOFFLINE_JETMET_BEAMHALO_ANALYZER_H (1)

//authors:  Ronny Remington, University of Florida
//date:  08/01/09

//Included Classes (semi-alphabetical)
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2D.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCRecHit2DCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCSegment.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/Cone.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCStatusDigiCollection.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutRecord.h"
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "DataFormats/METReco/interface/CSCHaloData.h"
#include "DataFormats/METReco/interface/EcalHaloData.h"
#include "DataFormats/METReco/interface/HcalHaloData.h"
#include "DataFormats/METReco/interface/GlobalHaloData.h"
#include "DataFormats/METReco/interface/BeamHaloSummary.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETFwd.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/MuonDetId/interface/CSCIndexer.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtra.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtraMap.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CSCGeometry/interface/CSCChamber.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include "Geometry/CSCGeometry/interface/CSCLayerGeometry.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "RecoMuon/MuonIdentification/interface/TimeMeasurementSequence.h"
#include "RecoMuon/TrackingTools/interface/MuonSegmentMatcher.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

//Root Classes

#include "TH1F.h"
#include "TH2F.h"
#include "TH1I.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TTree.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TString.h"
#include "TMath.h"
#include "TLorentzVector.h"
#include "TLegend.h"

//Standard C++ classes
#include <string>
#include <map>
#include <vector>
#include <utility>
#include <ostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <memory>
#include <iomanip>

class MuonServiceProxy;

class BeamHaloAnalyzer : public DQMEDAnalyzer {
public:
  explicit BeamHaloAnalyzer(const edm::ParameterSet&);
  ~BeamHaloAnalyzer() override;

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  edm::InputTag IT_L1MuGMTReadout;

  //RecHit Level
  edm::EDGetTokenT<CSCRecHit2DCollection> IT_CSCRecHit;
  edm::EDGetTokenT<EBRecHitCollection> IT_EBRecHit;
  edm::EDGetTokenT<EERecHitCollection> IT_EERecHit;
  edm::EDGetTokenT<ESRecHitCollection> IT_ESRecHit;
  edm::EDGetTokenT<HBHERecHitCollection> IT_HBHERecHit;
  edm::EDGetTokenT<HORecHitCollection> IT_HORecHit;
  edm::EDGetTokenT<HFRecHitCollection> IT_HFRecHit;

  //Higher Level Reco
  edm::EDGetTokenT<CSCSegmentCollection> IT_CSCSegment;
  edm::EDGetTokenT<reco::MuonCollection> IT_CollisionMuon;
  edm::EDGetTokenT<reco::MuonCollection> IT_CollisionStandAloneMuon;
  edm::EDGetTokenT<reco::MuonCollection> IT_BeamHaloMuon;
  edm::EDGetTokenT<reco::MuonCollection> IT_CosmicStandAloneMuon;
  edm::EDGetTokenT<reco::CaloMETCollection> IT_met;
  edm::EDGetTokenT<edm::View<reco::Candidate> > IT_CaloTower;
  edm::EDGetTokenT<reco::SuperClusterCollection> IT_SuperCluster;
  edm::EDGetTokenT<reco::PhotonCollection> IT_Photon;

  // Halo Data
  edm::EDGetTokenT<reco::CSCHaloData> IT_CSCHaloData;
  edm::EDGetTokenT<reco::EcalHaloData> IT_EcalHaloData;
  edm::EDGetTokenT<reco::HcalHaloData> IT_HcalHaloData;
  edm::EDGetTokenT<reco::GlobalHaloData> IT_GlobalHaloData;
  edm::EDGetTokenT<reco::BeamHaloSummary> IT_BeamHaloSummary;
  edm::EDGetTokenT<reco::MuonTimeExtraMap> IT_CSCTimeMapToken;

  edm::ESGetToken<CSCGeometry, MuonGeometryRecord> cscGeomToken_;

  //Output File
  std::string OutputFileName;
  std::string TextFileName;
  std::string FolderName;

  std::ofstream* out;

  double DumpMET;

  //Muon-Segment Matching
  MuonServiceProxy* TheService;
  MuonSegmentMatcher* TheMatcher;

  bool StandardDQM;

  MonitorElement* hEcalHaloData_PhiWedgeMultiplicity;
  MonitorElement* hEcalHaloData_PhiWedgeConstituents;
  MonitorElement* hEcalHaloData_PhiWedgeZDirectionConfidence;
  MonitorElement* hEcalHaloData_SuperClusterShowerShapes;
  MonitorElement* hEcalHaloData_SuperClusterEnergy;
  MonitorElement* hEcalHaloData_SuperClusterNHits;

  MonitorElement* hEcalHaloData_PhiWedgeEnergy;
  MonitorElement* hEcalHaloData_PhiWedgeMinTime;
  MonitorElement* hEcalHaloData_PhiWedgeMaxTime;
  MonitorElement* hEcalHaloData_PhiWedgeiPhi;
  MonitorElement* hEcalHaloData_PhiWedgePlusZDirectionConfidence;
  MonitorElement* hEcalHaloData_PhiWedgeMinVsMaxTime;
  MonitorElement* hEcalHaloData_SuperClusterPhiVsEta;

  MonitorElement* hHcalHaloData_PhiWedgeMultiplicity;
  MonitorElement* hHcalHaloData_PhiWedgeConstituents;
  MonitorElement* hHcalHaloData_PhiWedgeZDirectionConfidence;

  MonitorElement* hHcalHaloData_PhiWedgeEnergy;
  MonitorElement* hHcalHaloData_PhiWedgeiPhi;
  MonitorElement* hHcalHaloData_PhiWedgeMinTime;
  MonitorElement* hHcalHaloData_PhiWedgeMaxTime;
  MonitorElement* hHcalHaloData_PhiWedgePlusZDirectionConfidence;
  MonitorElement* hHcalHaloData_PhiWedgeMinVsMaxTime;

  MonitorElement* hCSCHaloData_TrackMultiplicity;
  MonitorElement* hCSCHaloData_TrackMultiplicityMEPlus;
  MonitorElement* hCSCHaloData_TrackMultiplicityMEMinus;
  MonitorElement* hCSCHaloData_InnerMostTrackHitR;
  MonitorElement* hCSCHaloData_InnerMostTrackHitPhi;
  MonitorElement* hCSCHaloData_L1HaloTriggersMEPlus;
  MonitorElement* hCSCHaloData_L1HaloTriggersMEMinus;
  MonitorElement* hCSCHaloData_L1HaloTriggers;
  MonitorElement* hCSCHaloData_HLHaloTriggers;
  MonitorElement* hCSCHaloData_NOutOfTimeTriggersvsL1HaloExists;
  MonitorElement* hCSCHaloData_NOutOfTimeTriggersMEPlus;
  MonitorElement* hCSCHaloData_NOutOfTimeTriggersMEMinus;
  MonitorElement* hCSCHaloData_NOutOfTimeTriggers;
  MonitorElement* hCSCHaloData_NOutOfTimeHits;
  MonitorElement* hCSCHaloData_NTracksSmalldT;
  MonitorElement* hCSCHaloData_NTracksSmallBeta;
  MonitorElement* hCSCHaloData_NTracksSmallBetaAndSmalldT;
  MonitorElement* hCSCHaloData_NTracksSmalldTvsNHaloTracks;

  MonitorElement* hCSCHaloData_InnerMostTrackHitXY;
  MonitorElement* hCSCHaloData_InnerMostTrackHitRPlusZ;
  MonitorElement* hCSCHaloData_InnerMostTrackHitRMinusZ;
  MonitorElement* hCSCHaloData_InnerMostTrackHitiPhi;

  MonitorElement* hCSCHaloData_SegmentdT;
  MonitorElement* hCSCHaloData_FreeInverseBeta;
  MonitorElement* hCSCHaloData_FreeInverseBetaVsSegmentdT;

  // MLR
  MonitorElement* hCSCHaloData_NFlatHaloSegments;
  MonitorElement* hCSCHaloData_SegmentsInBothEndcaps;
  MonitorElement* hCSCHaloData_NFlatSegmentsInBothEndcaps;
  // End MLR

  MonitorElement* hGlobalHaloData_MExCorrection;
  MonitorElement* hGlobalHaloData_MEyCorrection;
  MonitorElement* hGlobalHaloData_SumEtCorrection;
  MonitorElement* hGlobalHaloData_HaloCorrectedMET;
  MonitorElement* hGlobalHaloData_RawMETMinusHaloCorrectedMET;
  MonitorElement* hGlobalHaloData_RawMETOverSumEt;
  MonitorElement* hGlobalHaloData_MatchedHcalPhiWedgeMultiplicity;
  MonitorElement* hGlobalHaloData_MatchedHcalPhiWedgeEnergy;
  MonitorElement* hGlobalHaloData_MatchedHcalPhiWedgeConstituents;
  MonitorElement* hGlobalHaloData_MatchedHcalPhiWedgeiPhi;
  MonitorElement* hGlobalHaloData_MatchedHcalPhiWedgeMinTime;
  MonitorElement* hGlobalHaloData_MatchedHcalPhiWedgeMaxTime;
  MonitorElement* hGlobalHaloData_MatchedHcalPhiWedgeZDirectionConfidence;
  MonitorElement* hGlobalHaloData_MatchedEcalPhiWedgeMultiplicity;
  MonitorElement* hGlobalHaloData_MatchedEcalPhiWedgeEnergy;
  MonitorElement* hGlobalHaloData_MatchedEcalPhiWedgeConstituents;
  MonitorElement* hGlobalHaloData_MatchedEcalPhiWedgeiPhi;
  MonitorElement* hGlobalHaloData_MatchedEcalPhiWedgeMinTime;
  MonitorElement* hGlobalHaloData_MatchedEcalPhiWedgeMaxTime;
  MonitorElement* hGlobalHaloData_MatchedEcalPhiWedgeZDirectionConfidence;

  MonitorElement* hBeamHaloSummary_Id;

  MonitorElement* hBeamHaloSummary_BXN;
  MonitorElement* hExtra_InnerMostTrackHitR;
  MonitorElement* hExtra_CSCActivityWithMET;
  MonitorElement* hExtra_HcalToF;
  MonitorElement* hExtra_HcalToF_HaloId;
  MonitorElement* hExtra_EcalToF;
  MonitorElement* hExtra_EcalToF_HaloId;
  MonitorElement* hExtra_CSCTrackInnerOuterDPhi;
  MonitorElement* hExtra_CSCTrackInnerOuterDEta;
  MonitorElement* hExtra_CSCTrackChi2Ndof;
  MonitorElement* hExtra_CSCTrackNHits;
  MonitorElement* hExtra_InnerMostTrackHitXY;
  MonitorElement* hExtra_InnerMostTrackHitRPlusZ;
  MonitorElement* hExtra_InnerMostTrackHitRMinusZ;
  MonitorElement* hExtra_InnerMostTrackHitiPhi;
  MonitorElement* hExtra_InnerMostTrackHitPhi;
  MonitorElement* hExtra_BXN;
};

#endif
