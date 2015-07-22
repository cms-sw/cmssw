#ifndef CalibrationIsolatedParticlesIsolatedTracksHcalScale_h
#define CalibrationIsolatedParticlesIsolatedTracksHcalScale_h

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
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetup.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"

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

#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenuFwd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"

#include "Calibration/IsolatedParticles/interface/FindCaloHit.h"
#include "Calibration/IsolatedParticles/interface/eECALMatrix.h"
#include "Calibration/IsolatedParticles/interface/eHCALMatrix.h"
#include "Calibration/IsolatedParticles/interface/MatchingSimTrack.h"
#include "Calibration/IsolatedParticles/interface/CaloSimInfo.h"
#include "Calibration/IsolatedParticles/interface/TrackSelection.h"

// root objects
#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TDirectory.h"
#include "TTree.h"

class IsolatedTracksHcalScale : public edm::EDAnalyzer {
  
public:
  explicit IsolatedTracksHcalScale(const edm::ParameterSet&);
  ~IsolatedTracksHcalScale();
  
private:
  //void   beginJob(const edm::EventSetup&) ;
  void   beginJob() ;
  void   analyze(const edm::Event&, const edm::EventSetup&);
  void   endJob() ;

  void   clearTreeVectors();  
  
private:
  
  bool        initL1, doMC;
  int         myverbose;
  std::string theTrackQuality, minQuality;
  spr::trackSelectionParameters selectionParameters;
  double      a_mipR, a_coneR, a_charIsoR, a_neutIsoR;
  double      tMinE_, tMaxE_;

  TrackerHitAssociator::Config trackerHitAssociatorConfig_;

  edm::EDGetTokenT<reco::TrackCollection>   tok_genTrack_;
  edm::EDGetTokenT<reco::VertexCollection>  tok_recVtx_;
  edm::EDGetTokenT<reco::BeamSpot>          tok_bs_;
  edm::EDGetTokenT<EcalRecHitCollection>    tok_EB_;
  edm::EDGetTokenT<EcalRecHitCollection>    tok_EE_;
  edm::EDGetTokenT<HBHERecHitCollection>    tok_hbhe_;

  edm::EDGetTokenT<edm::SimTrackContainer>  tok_simTk_;
  edm::EDGetTokenT<edm::SimVertexContainer> tok_simVtx_;
  edm::EDGetTokenT<edm::PCaloHitContainer>  tok_caloEB_;
  edm::EDGetTokenT<edm::PCaloHitContainer>  tok_caloEE_;
  edm::EDGetTokenT<edm::PCaloHitContainer>  tok_caloHH_;

  int    nEventProc;
  const MagneticField *bField;

  double genPartEtaBins[4];

  static const size_t NEtaBins = 3;
  
  TTree* tree;

  int  t_nTracks, t_RunNo, t_EvtNo, t_Lumi, t_Bunch;
  std::vector<double> *t_trackP,        *t_trackPt,        *t_trackEta,      *t_trackPhi;
  std::vector<double> *t_trackHcalEta,  *t_trackHcalPhi,   *t_eHCALDR;   
  std::vector<double> *t_hCone,         *t_conehmaxNearP,  *t_eMipDR,        *t_eECALDR;
  std::vector<double> *t_e11x11_20Sig,  *t_e15x15_20Sig;
  std::vector<double> *t_eMipDR_1,      *t_eECALDR_1,      *t_eMipDR_2,      *t_eECALDR_2;
  std::vector<double> *t_hConeHB,       *t_eHCALDRHB;
  std::vector<double> *t_hsimInfoMatched,  *t_hsimInfoRest,     *t_hsimInfoPhoton;
  std::vector<double> *t_hsimInfoNeutHad,  *t_hsimInfoCharHad,  *t_hsimInfoPdgMatched;
  std::vector<double> *t_hsimInfoTotal,    *t_hsim;
  std::vector<int>    *t_hsimInfoNMatched, *t_hsimInfoNTotal,   *t_hsimInfoNNeutHad;
  std::vector<int>    *t_hsimInfoNCharHad, *t_hsimInfoNPhoton,  *t_hsimInfoNRest;
  std::vector<int>    *t_nSimHits;
  edm::Service<TFileService> fs;
};

#endif
