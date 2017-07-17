#ifndef CalibrationIsolatedParticlesIsolatedTracksNxN_h
#define CalibrationIsolatedParticlesIsolatedTracksNxN_h

// system include files
#include <memory>
#include <cmath>
#include <string>
#include <map>
#include <vector>

// user include files
#include <Math/GenVector/VectorUtil.h>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// TFile Service
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Candidate/interface/Candidate.h"

// muons and tracks
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
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

// SimHit
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
//simtrack
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

// track associator
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"

// tracker hit associator
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"

// ecal / hcal
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"

#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

//L1 trigger Menus etc
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"
#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"

#include "Calibration/IsolatedParticles/interface/FindCaloHit.h"
#include "Calibration/IsolatedParticles/interface/eECALMatrix.h"
#include "Calibration/IsolatedParticles/interface/eHCALMatrix.h"
#include "Calibration/IsolatedParticles/interface/MatchingSimTrack.h"
#include "Calibration/IsolatedParticles/interface/CaloSimInfo.h"

// root objects
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TDirectory.h"
#include "TTree.h"

class IsolatedTracksNxN : public edm::EDAnalyzer {

public:
  explicit IsolatedTracksNxN(const edm::ParameterSet&);
  ~IsolatedTracksNxN();
  
private:
  //void   beginJob(const edm::EventSetup&) ;
  void   beginJob() ;
  void   analyze(const edm::Event&, const edm::EventSetup&);
  void   endJob() ;

  void   printTrack(const reco::Track* pTrack);

  void   BookHistograms();

  double DeltaPhi(double v1, double v2);
  double DeltaR(double eta1, double phi1, double eta2, double phi2);


  void clearTreeVectors();  
  
private:

  L1GtUtils m_l1GtUtils;

  bool initL1, doMC, writeAllTracks;
  static const size_t nL1BitsMax=128;

  // map of trig bit, algo name and num events passed
  std::map< std::pair<unsigned int,std::string>, int> l1AlgoMap;
  std::vector<unsigned int> m_triggerMaskAlgoTrig;

  double pvTracksPtMin_;
  bool   debugL1Info_, L1TriggerAlgoInfo_;
  int    debugTrks_;
  bool   printTrkHitPattern_;
  int    myverbose_;

  TrackerHitAssociator::Config trackerHitAssociatorConfig_;

  edm::EDGetTokenT<l1extra::L1JetParticleCollection>  tok_L1extTauJet_;
  edm::EDGetTokenT<l1extra::L1JetParticleCollection>  tok_L1extCenJet_;
  edm::EDGetTokenT<l1extra::L1JetParticleCollection>  tok_L1extFwdJet_;

  edm::EDGetTokenT<l1extra::L1MuonParticleCollection> tok_L1extMu_;
  edm::EDGetTokenT<l1extra::L1EmParticleCollection>   tok_L1extIsoEm_;
  edm::EDGetTokenT<l1extra::L1EmParticleCollection>   tok_L1extNoIsoEm_;

  edm::EDGetTokenT<reco::CaloJetCollection>           tok_jets_;
  edm::EDGetTokenT<HBHERecHitCollection>              tok_hbhe_;

  edm::EDGetTokenT<reco::TrackCollection>             tok_genTrack_;
  edm::EDGetTokenT<reco::VertexCollection>            tok_recVtx_;
  edm::EDGetTokenT<reco::BeamSpot>                    tok_bs_;

  edm::EDGetTokenT<EcalRecHitCollection>              tok_EB_;
  edm::EDGetTokenT<EcalRecHitCollection>              tok_EE_;
  edm::EDGetTokenT<edm::SimTrackContainer>            tok_simTk_;
  edm::EDGetTokenT<edm::SimVertexContainer>           tok_simVtx_;
  edm::EDGetTokenT<edm::PCaloHitContainer>            tok_caloEB_;
  edm::EDGetTokenT<edm::PCaloHitContainer>            tok_caloEE_;
  edm::EDGetTokenT<edm::PCaloHitContainer>            tok_caloHH_;

  double minTrackP_, maxTrackEta_;
  double tMinE_, tMaxE_, tMinH_, tMaxH_;
  int    nEventProc;

  const MagneticField *bField;

  double genPartPBins[16], genPartEtaBins[4];

  static const size_t NPBins   = 15;
  static const size_t NEtaBins = 3;
  
  TH1F *h_maxNearP15x15[NPBins][NEtaBins],
       *h_maxNearP21x21[NPBins][NEtaBins],
       *h_maxNearP25x25[NPBins][NEtaBins],
       *h_maxNearP31x31[NPBins][NEtaBins];

  TH1I *h_L1AlgoNames;
  TH1F *h_PVTracksWt;

  TH1F *h_nTracks;

  TH1F *h_recPt_0,    *h_recP_0,    *h_recEta_0, *h_recPhi_0;
  TH2F *h_recEtaPt_0, *h_recEtaP_0;

  TH1F *h_recPt_1,    *h_recP_1,    *h_recEta_1, *h_recPhi_1;
  TH2F *h_recEtaPt_1, *h_recEtaP_1;

  TH1F *h_recPt_2,    *h_recP_2,    *h_recEta_2, *h_recPhi_2;
  TH2F *h_recEtaPt_2, *h_recEtaP_2;

  
 
  TTree* tree;

  int    t_nTracks;

  int t_RunNo, t_EvtNo, t_Lumi, t_Bunch;
  std::vector<std::string> *t_L1AlgoNames;
  std::vector<int>         *t_L1PreScale;
  int                      t_L1Decision[128];

  std::vector<double> *t_PVx, *t_PVy, *t_PVz, *t_PVTracksSumPt;
  std::vector<double> *t_PVTracksSumPtWt, *t_PVTracksSumPtHP, *t_PVTracksSumPtHPWt;
  std::vector<int>    *t_PVisValid, *t_PVNTracks, *t_PVNTracksWt, *t_PVndof;
  std::vector<int>    *t_PVNTracksHP, *t_PVNTracksHPWt;

  std::vector<double> *t_L1CenJetPt,    *t_L1CenJetEta,    *t_L1CenJetPhi;
  std::vector<double> *t_L1FwdJetPt,    *t_L1FwdJetEta,    *t_L1FwdJetPhi;
  std::vector<double> *t_L1TauJetPt,    *t_L1TauJetEta,    *t_L1TauJetPhi;
  std::vector<double> *t_L1MuonPt,      *t_L1MuonEta,      *t_L1MuonPhi;
  std::vector<double> *t_L1IsoEMPt,     *t_L1IsoEMEta,     *t_L1IsoEMPhi;
  std::vector<double> *t_L1NonIsoEMPt,  *t_L1NonIsoEMEta,  *t_L1NonIsoEMPhi;
  std::vector<double> *t_L1METPt,       *t_L1METEta,       *t_L1METPhi;

  std::vector<double> *t_jetPt,         *t_jetEta,         *t_jetPhi;
  std::vector<double> *t_nTrksJetCalo,  *t_nTrksJetVtx;

  std::vector<double> *t_trackPAll,     *t_trackEtaAll,    *t_trackPhiAll,  *t_trackPdgIdAll;
  std::vector<double> *t_trackPtAll;
  std::vector<double> *t_trackDxyAll,   *t_trackDzAll,     *t_trackDxyPVAll,*t_trackDzPVAll, *t_trackChiSqAll;

  std::vector<double> *t_trackP,        *t_trackPt,        *t_trackEta,      *t_trackPhi;
  std::vector<double> *t_trackEcalEta,  *t_trackEcalPhi,   *t_trackHcalEta,  *t_trackHcalPhi;   
  std::vector<double> *t_trackDxy,      *t_trackDxyBS,     *t_trackDz,       *t_trackDzBS;  
  std::vector<double> *t_trackDxyPV,    *t_trackDzPV;
  std::vector<double> *t_trackChiSq;
  std::vector<int>    *t_trackPVIdx;

  std::vector<int>    *t_NLayersCrossed,    *t_trackNOuterHits;
  std::vector<int>    *t_trackHitsTOB,      *t_trackHitsTEC;
  std::vector<int>    *t_trackHitInMissTOB, *t_trackHitInMissTEC,  *t_trackHitInMissTIB,  *t_trackHitInMissTID,  *t_trackHitInMissTIBTID;
  std::vector<int>    *t_trackHitOutMissTOB,*t_trackHitOutMissTEC, *t_trackHitOutMissTIB, *t_trackHitOutMissTID, *t_trackHitOutMissTOBTEC;
  std::vector<int>    *t_trackHitInMeasTOB, *t_trackHitInMeasTEC,  *t_trackHitInMeasTIB,  *t_trackHitInMeasTID;
  std::vector<int>    *t_trackHitOutMeasTOB,*t_trackHitOutMeasTEC, *t_trackHitOutMeasTIB, *t_trackHitOutMeasTID;
  std::vector<double> *t_trackOutPosOutHitDr, *t_trackL;

  std::vector<double> *t_maxNearP31x31;
  std::vector<double> *t_maxNearP21x21;

  std::vector<int>    *t_ecalSpike11x11;
  std::vector<double> *t_e7x7,       *t_e9x9,       *t_e11x11,       *t_e15x15;
  std::vector<double> *t_e7x7_10Sig, *t_e9x9_10Sig, *t_e11x11_10Sig, *t_e15x15_10Sig;
  std::vector<double> *t_e7x7_15Sig, *t_e9x9_15Sig, *t_e11x11_15Sig, *t_e15x15_15Sig;
  std::vector<double> *t_e7x7_20Sig, *t_e9x9_20Sig, *t_e11x11_20Sig, *t_e15x15_20Sig;
  std::vector<double> *t_e7x7_25Sig, *t_e9x9_25Sig, *t_e11x11_25Sig, *t_e15x15_25Sig;
  std::vector<double> *t_e7x7_30Sig, *t_e9x9_30Sig, *t_e11x11_30Sig, *t_e15x15_30Sig;

  std::vector<double> *t_esimPdgId,         *t_simTrackP;

  std::vector<double> *t_trkEcalEne;

  std::vector<double> *t_esim7x7,       *t_esim9x9,         *t_esim11x11,        *t_esim15x15;
  std::vector<double> *t_esim7x7Matched, *t_esim9x9Matched, *t_esim11x11Matched, *t_esim15x15Matched;
  std::vector<double> *t_esim7x7Rest,    *t_esim9x9Rest,    *t_esim11x11Rest,    *t_esim15x15Rest;
  std::vector<double> *t_esim7x7Photon,  *t_esim9x9Photon,  *t_esim11x11Photon,  *t_esim15x15Photon;
  std::vector<double> *t_esim7x7NeutHad, *t_esim9x9NeutHad, *t_esim11x11NeutHad, *t_esim15x15NeutHad;
  std::vector<double> *t_esim7x7CharHad, *t_esim9x9CharHad, *t_esim11x11CharHad, *t_esim15x15CharHad;

  std::vector<double> *t_maxNearHcalP3x3,   *t_maxNearHcalP5x5,   *t_maxNearHcalP7x7;
  std::vector<double> *t_h3x3,              *t_h5x5,              *t_h7x7;
  std::vector<double> *t_h3x3Sig,           *t_h5x5Sig,           *t_h7x7Sig;
  std::vector<int>    *t_infoHcal;

  std::vector<double> *t_trkHcalEne;
  std::vector<double> *t_hsim3x3,           *t_hsim5x5,           *t_hsim7x7;
  std::vector<double> *t_hsim3x3Matched,    *t_hsim5x5Matched,    *t_hsim7x7Matched;
  std::vector<double> *t_hsim3x3Rest,       *t_hsim5x5Rest,       *t_hsim7x7Rest;
  std::vector<double> *t_hsim3x3Photon,     *t_hsim5x5Photon,     *t_hsim7x7Photon;
  std::vector<double> *t_hsim3x3NeutHad,    *t_hsim5x5NeutHad,    *t_hsim7x7NeutHad;
  std::vector<double> *t_hsim3x3CharHad,    *t_hsim5x5CharHad,    *t_hsim7x7CharHad;

  edm::Service<TFileService> fs;
  int                 nbad;
};

#endif
