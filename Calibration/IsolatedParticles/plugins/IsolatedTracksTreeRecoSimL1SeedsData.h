#ifndef IsolatedTracksTreeRecoSimL1SeedsData_h
#define IsolatedTracksTreeRecoSimL1SeedsData_h

// system include files
#include <memory>
#include <cmath>
#include <string>
#include <map>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <Math/GenVector/VectorUtil.h>
#include "DataFormats/Common/interface/Ref.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Candidate/interface/Candidate.h"

// muons and tracks
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
// track associator
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
// Vertices
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
// SimHit
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
//simtrack
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
// ecal / hcal
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"

//L1 objects
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"

// Jets in the event
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/JetExtendedAssociation.h"

// TFile Service
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "PhysicsTools/UtilAlgos/interface/TFileService.h"

#include "Calibration/IsolatedParticles/interface/FindCaloHit.h"
#include "Calibration/IsolatedParticles/interface/eECALMatrix.h"
#include "Calibration/IsolatedParticles/interface/eHCALMatrix.h"
#include "Calibration/IsolatedParticles/interface/MatchingSimTrack.h"
#include "Calibration/IsolatedParticles/interface/CaloSimInfo.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/Math/interface/Point3D.h"

// root objects
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TDirectory.h"
#include "TTree.h"

#include <cmath>

class IsolatedTracksTreeRecoSimL1SeedsData : public edm::EDAnalyzer {
 public:
  explicit IsolatedTracksTreeRecoSimL1SeedsData(const edm::ParameterSet&);
  ~IsolatedTracksTreeRecoSimL1SeedsData();
  
 private:
  void   beginJob(const edm::EventSetup&) ;
  void   analyze(const edm::Event&, const edm::EventSetup&);
  void   endJob() ;

  void   printTrack(const reco::Track* pTrack);

  void   BookHistograms();

  double DeltaPhi(double v1, double v2);
  double DeltaR(double eta1, double phi1, double eta2, double phi2);

  int    chargeIsolation(CaloNavigator<DetId>& navigator,const DetId anyCell, int deta, int dphi);
  double chargeIsolation(const edm::Event& iEvent, const edm::EventSetup& iSetup, 
			 CaloNavigator<DetId>& navigator, 
			 reco::TrackCollection::const_iterator trkItr, 
			 edm::Handle<reco::TrackCollection> trkCollection, 
			 edm::Handle<EcalRecHitCollection>& barrelRecHitsHandle, 
			 edm::Handle<EcalRecHitCollection>& endcapRecHitsHandle, 
			 const CaloSubdetectorGeometry* gEB, const CaloSubdetectorGeometry* gEE, 
			 int ieta, int iphi);

  double chargeIsolationNew(const edm::Event& iEvent, const edm::EventSetup& iSetup, 
			    const DetId& coreDet,
			    reco::TrackCollection::const_iterator trkItr, 
			    edm::Handle<reco::TrackCollection> trkCollection, 
			    edm::Handle<EcalRecHitCollection>& barrelRecHitsHandle, 
			    edm::Handle<EcalRecHitCollection>& endcapRecHitsHandle, 
			    const CaloGeometry* geo, const CaloTopology* caloTopology,
			    int ieta, int iphi);
  int    chargeIsolationNew(std::vector<DetId>& vdets, const DetId anyCell) ;

  double chargeIsolationHcal(const edm::Event& iEvent, const edm::EventSetup& iSetup, 
			     reco::TrackCollection::const_iterator trkItr, 
			     edm::Handle<reco::TrackCollection> trkCollection,
			     const DetId ClosestCell, const HcalTopology* topology, 
			     const CaloSubdetectorGeometry* gHB,
			     int ieta, int iphi, bool debug=false);

  void clearTreeVectors();  

  bool initL1;
  static const size_t nL1BitsMax=128;
  std::string algoBitToName[nL1BitsMax];
  std::map <std::string,bool> l1TriggerMap;
  std::map<std::string,bool>::iterator trig_iter;

  double pvTracksPtMin_;
  bool   debugL1Info_;
  int    debugTrks_;
  bool   printTrkHitPattern_;
  int    myverbose_;
  edm::InputTag L1extraTauJetSource_,  L1extraCenJetSource_,    L1extraFwdJetSource_;
  edm::InputTag L1extraMuonSource_,    L1extraIsoEmSource_,     L1extraNonIsoEmSource_;
  edm::InputTag L1GTReadoutRcdSource_, L1GTObjectMapRcdSource_;
  edm::InputTag JetExtender_, JetSrc_;

  double minTrackP_, maxTrackEta_;

  int    nEventProc;

  double genPartPBins[22], genPartEtaBins[5];

  // track associator to detector parameters 
  TrackAssociatorParameters parameters_;
  mutable TrackDetectorAssociator* trackAssociator_;

  const MagneticField *bField;

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
  std::vector<double> *t_PVx,*t_PVy,*t_PVz, *t_PVisValid, *t_PVNTracks, *t_PVNTracksWt, *t_PVTracksSumPt, *t_PVTracksSumPtWt ;
  std::vector<double> *t_PVNTracksHP, *t_PVNTracksHPWt, *t_PVTracksSumPtHP, *t_PVTracksSumPtHPWt ;

  std::vector<int>    *t_L1Decision;
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

  std::vector<double> *t_trackP,        *t_trackPt,        *t_trackEta,      *t_trackPhi;
  std::vector<double> *t_trackDxy,      *t_trackDxyBS,     *t_trackDz,       *t_trackDzBS;  
  std::vector<double> *t_trackDxyPV,    *t_trackDzPV;
  std::vector<double> *t_trackChiSq;
  std::vector<int>    *t_trackPVIdx;

  std::vector<int>    *t_NLayersCrossed,*t_trackNOuterHits;

  std::vector<double> *t_maxNearP31x31;
  std::vector<double> *t_maxNearP25x25;
  std::vector<double> *t_maxNearP21x21;
  std::vector<double> *t_maxNearP15x15;
  std::vector<double> *t_maxNearP13x13;
  std::vector<double> *t_maxNearP11x11;
  std::vector<double> *t_maxNearP9x9;
  std::vector<double> *t_maxNearP7x7;

  std::vector<double> *t_e3x3,              *t_e5x5,              *t_e7x7,              *t_e9x9,             *t_e11x11; 
  std::vector<double> *t_e13x13,            *t_e15x15,            *t_e21x21,            *t_e25x25,           *t_e31x31;

  std::vector<double> *t_esimPdgId,         *t_simTrackP;

  std::vector<double> *t_trkEcalEne;

  std::vector<double> *t_esim3x3,           *t_esim5x5,           *t_esim7x7,           *t_esim9x9,          *t_esim11x11; 
  std::vector<double> *t_esim13x13,         *t_esim15x15,         *t_esim21x21,         *t_esim25x25,        *t_esim31x31;

  std::vector<double> *t_esim3x3Matched,    *t_esim5x5Matched,    *t_esim7x7Matched,    *t_esim9x9Matched,   *t_esim11x11Matched; 
  std::vector<double> *t_esim13x13Matched,  *t_esim15x15Matched,  *t_esim21x21Matched,  *t_esim25x25Matched, *t_esim31x31Matched;

  std::vector<double> *t_esim3x3Rest,       *t_esim5x5Rest,       *t_esim7x7Rest,       *t_esim9x9Rest,      *t_esim11x11Rest; 
  std::vector<double> *t_esim13x13Rest,     *t_esim15x15Rest,     *t_esim21x21Rest,     *t_esim25x25Rest,    *t_esim31x31Rest;

  std::vector<double> *t_esim3x3Photon,     *t_esim5x5Photon,     *t_esim7x7Photon,     *t_esim9x9Photon,   *t_esim11x11Photon; 
  std::vector<double> *t_esim13x13Photon,   *t_esim15x15Photon,   *t_esim21x21Photon,   *t_esim25x25Photon, *t_esim31x31Photon;

  std::vector<double> *t_esim3x3NeutHad,    *t_esim5x5NeutHad,    *t_esim7x7NeutHad,    *t_esim9x9NeutHad,   *t_esim11x11NeutHad; 
  std::vector<double> *t_esim13x13NeutHad,  *t_esim15x15NeutHad,  *t_esim21x21NeutHad,  *t_esim25x25NeutHad, *t_esim31x31NeutHad;

  std::vector<double> *t_esim3x3CharHad,    *t_esim5x5CharHad,    *t_esim7x7CharHad,    *t_esim9x9CharHad,   *t_esim11x11CharHad; 
  std::vector<double> *t_esim13x13CharHad,  *t_esim15x15CharHad,  *t_esim21x21CharHad,  *t_esim25x25CharHad, *t_esim31x31CharHad;


  std::vector<double> *t_maxNearHcalP3x3,   *t_maxNearHcalP5x5,   *t_maxNearHcalP7x7;
  std::vector<double> *t_h3x3,              *t_h5x5,              *t_h7x7;
  std::vector<int>    *t_infoHcal;

  std::vector<double> *t_trkHcalEne;
  std::vector<double> *t_hsim3x3,           *t_hsim5x5,           *t_hsim7x7;
  std::vector<double> *t_hsim3x3Matched,    *t_hsim5x5Matched,    *t_hsim7x7Matched;
  std::vector<double> *t_hsim3x3Rest,       *t_hsim5x5Rest,       *t_hsim7x7Rest;
  std::vector<double> *t_hsim3x3Photon,     *t_hsim5x5Photon,     *t_hsim7x7Photon;
  std::vector<double> *t_hsim3x3NeutHad,    *t_hsim5x5NeutHad,    *t_hsim7x7NeutHad;
  std::vector<double> *t_hsim3x3CharHad,    *t_hsim5x5CharHad,    *t_hsim7x7CharHad;

  edm::Service<TFileService> fs;
};

#endif
