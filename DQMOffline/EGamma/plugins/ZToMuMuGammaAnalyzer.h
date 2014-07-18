#ifndef ZToMuMuGammaAnalyzer_H
#define ZToMuMuGammaAnalyzer_H

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
//
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
// DataFormats
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

/// EgammaCoreTools
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalEtaPhiRegion.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterTools.h"

#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"

// Geometry
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalEndcapTopology.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
//
#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TTree.h"
#include "TVector3.h"
#include "TProfile.h"
//

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

//
//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"



#include <vector>
#include <string>

/** \class ZToMuMuGammaAnalyzer
 **  
 **
 **  $Id: ZToMuMuGammaAnalyzer
 **  authors: 
 **   Nancy Marinelli, U. of Notre Dame, US  
 **   Jamie Antonelli, U. of Notre Dame, US
 **   Nathan Kellams,  U. of Notre Dame, US
 **     
 ***/


// forward declarations
class TFile;
class TH1F;
class TH2F;
class TProfile;
class TTree;
class SimVertex;
class SimTrack;


class ZToMuMuGammaAnalyzer  : public edm::EDAnalyzer
{


 public:
   
  //
  explicit ZToMuMuGammaAnalyzer( const edm::ParameterSet& ) ;
  virtual ~ZToMuMuGammaAnalyzer();
                                   
  virtual void analyze( const edm::Event&, const edm::EventSetup& ) ;
  virtual void beginJob() ;
  virtual void endJob() ;
  virtual void endRun(const edm::Run& , const edm::EventSetup& ) ;
  
      
 private:
  edm::EDGetTokenT<std::vector<reco::Photon> > photon_token_;
  edm::EDGetTokenT<std::vector<reco::Muon> > muon_token_;
  edm::EDGetTokenT<edm::ValueMap<bool> > PhotonIDLoose_token_;
  edm::EDGetTokenT<edm::ValueMap<bool> > PhotonIDTight_token_;
  edm::EDGetTokenT<edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> > > barrelRecHit_token_;
  edm::EDGetTokenT<edm::SortedCollection<EcalRecHit,edm::StrictWeakOrdering<EcalRecHit> > > endcapRecHit_token_;
  edm::EDGetTokenT<trigger::TriggerEvent> triggerEvent_token_;
  edm::EDGetTokenT<reco::BeamSpot> beamSpot_token_;
  edm::EDGetTokenT<reco::PFCandidateCollection> pfCandidates_;
  edm::EDGetTokenT<reco::VertexCollection> offline_pvToken_;
  std::string   valueMapPhoPFCandIso_ ;
  

  std::string fName_;
  int verbosity_;

  bool useTriggerFiltering_;
  bool splitHistosEBEE_;
  bool use2DHistos_;
  bool makeProfiles_;
  unsigned int prescaleFactor_;
  bool standAlone_;
  std::string outputFileName_;
  edm::ParameterSet parameters_;
  bool isHeavyIon_;
  DQMStore *dbe_;
  std::stringstream currentFolder_;
  int nEvt_;
  int nEntry_;


  // muon selection
  float muonMinPt_;
  int minPixStripHits_;
  float muonMaxChi2_;
  float muonMaxDxy_;
  int muonMatches_;
  int validPixHits_;
  int validMuonHits_;
  float muonTrackIso_;
  float muonTightEta_;
  // dimuon selection
  float minMumuInvMass_;
  float maxMumuInvMass_;
  // photon selection
  float photonMinEt_;
  float photonMaxEta_;
  float photonTrackIso_;
  
  // mu mu gamma selection
  float nearMuonDr_;
  float nearMuonHcalIso_;
  float farMuonEcalIso_;
  float farMuonTrackIso_;
  float farMuonMinPt_;
  float minMumuGammaInvMass_;
  float maxMumuGammaInvMass_;

  float mumuInvMass( const reco::Muon & m1,const reco::Muon & m2 ) ;
  float mumuGammaInvMass(const reco::Muon & mu1,const reco::Muon & mu2, const reco::PhotonRef& pho );
  bool  basicMuonSelection (  const reco::Muon & m );
  bool  muonSelection (  const reco::Muon & m,  const reco::BeamSpot& bs );
  bool  photonSelection (  const reco::PhotonRef & p );


  MonitorElement* h_nRecoVtx_;
  ///photon histos
  MonitorElement* h1_mumuInvMass_[3];
  MonitorElement* h1_mumuGammaInvMass_[3];

  MonitorElement* h_phoE_[3];
  MonitorElement* h_phoSigmaEoverE_[3];
  MonitorElement* p_phoSigmaEoverEVsNVtx_[3];
  MonitorElement* h_phoEt_[3];

  MonitorElement* h_nPho_[3];

  MonitorElement* h_phoEta_[3];
  MonitorElement* h_phoPhi_[3];
  MonitorElement* h_scEta_[3];
  MonitorElement* h_scPhi_[3];

  MonitorElement* h_r9_[3];
  MonitorElement* h2_r9VsEt_[3];
  MonitorElement* p_r9VsEt_[3];
  MonitorElement* h2_r9VsEta_[3];
  MonitorElement* p_r9VsEta_[3];

  MonitorElement* h_e1x5_[3];
  MonitorElement* h2_e1x5VsEta_[3];
  MonitorElement* p_e1x5VsEta_[3];
  MonitorElement* h2_e1x5VsEt_[3];
  MonitorElement* p_e1x5VsEt_[3];
        
  MonitorElement* h_e2x5_[3];
  MonitorElement* h2_e2x5VsEta_[3];
  MonitorElement* p_e2x5VsEta_[3];
  MonitorElement* h2_e2x5VsEt_[3];
  MonitorElement* p_e2x5VsEt_[3];
  
  MonitorElement* h_r1x5_[3];
  MonitorElement* h2_r1x5VsEta_[3];
  MonitorElement* p_r1x5VsEta_[3];
  MonitorElement* h2_r1x5VsEt_[3];
  MonitorElement* p_r1x5VsEt_[3];
        
  MonitorElement* h_r2x5_[3];
  MonitorElement* h2_r2x5VsEta_[3];
  MonitorElement* p_r2x5VsEta_[3];
  MonitorElement* h2_r2x5VsEt_[3];
  MonitorElement* p_r2x5VsEt_[3];
        
  MonitorElement* h_phoSigmaIetaIeta_[3];
  MonitorElement* h2_sigmaIetaIetaVsEta_[3]; 
  MonitorElement* p_sigmaIetaIetaVsEta_[3];

  MonitorElement* h_nTrackIsolSolid_[3];     
  MonitorElement* h2_nTrackIsolSolidVsEt_[3]; 
  MonitorElement* p_nTrackIsolSolidVsEt_[3]; 
  MonitorElement* h2_nTrackIsolSolidVsEta_[3];
  MonitorElement* p_nTrackIsolSolidVsEta_[3];
  
  MonitorElement* h_nTrackIsolHollow_[3];    
  MonitorElement* h2_nTrackIsolHollowVsEt_[3]; 
  MonitorElement* p_nTrackIsolHollowVsEt_[3]; 
  MonitorElement* h2_nTrackIsolHollowVsEta_[3];
  MonitorElement* p_nTrackIsolHollowVsEta_[3];
  
  MonitorElement* h_trackPtSumSolid_[3];      
  MonitorElement* h2_trackPtSumSolidVsEt_[3];  
  MonitorElement* p_trackPtSumSolidVsEt_[3];  
  MonitorElement* h2_trackPtSumSolidVsEta_[3]; 
  MonitorElement* p_trackPtSumSolidVsEta_[3]; 
 
  MonitorElement* h_trackPtSumHollow_[3];     
  MonitorElement* h2_trackPtSumHollowVsEt_[3]; 
  MonitorElement* p_trackPtSumHollowVsEt_[3]; 
  MonitorElement* h2_trackPtSumHollowVsEta_[3]; 
  MonitorElement* p_trackPtSumHollowVsEta_[3]; 

  MonitorElement* h_ecalSum_[3];    
  MonitorElement* h2_ecalSumVsEt_[3];  
  MonitorElement* p_ecalSumVsEt_[3];  
  MonitorElement* h2_ecalSumVsEta_[3]; 
  MonitorElement* p_ecalSumVsEta_[3]; 
  
  MonitorElement* h_hcalSum_[3];      
  MonitorElement* h2_hcalSumVsEt_[3];  
  MonitorElement* p_hcalSumVsEt_[3];  
  MonitorElement* h2_hcalSumVsEta_[3]; 
  MonitorElement* p_hcalSumVsEta_[3]; 
  
  MonitorElement* h_hOverE_[3];       
  MonitorElement* p_hOverEVsEt_[3];   
  MonitorElement* p_hOverEVsEta_[3];  
  MonitorElement* h_h1OverE_[3];      
  MonitorElement* h_h2OverE_[3];      

  MonitorElement* h_newhOverE_[3];
  MonitorElement* p_newhOverEVsEta_[3];
  MonitorElement* p_newhOverEVsEt_[3];
  // Information from Particle Flow
  // Isolation
  MonitorElement* h_chHadIso_[3];
  MonitorElement* h_nHadIso_[3];
  MonitorElement* h_phoIso_[3];
  // Identification
  MonitorElement* h_nCluOutsideMustache_[3];
  MonitorElement* h_etOutsideMustache_[3];
  MonitorElement* h_pfMva_[3];
  //// particle based isolation from ValueMap
  MonitorElement* h_dRPhoPFcand_ChHad_Cleaned_[3];
  MonitorElement* h_dRPhoPFcand_NeuHad_Cleaned_[3];
  MonitorElement* h_dRPhoPFcand_Pho_Cleaned_[3];
  MonitorElement* h_dRPhoPFcand_ChHad_unCleaned_[3];
  MonitorElement* h_dRPhoPFcand_NeuHad_unCleaned_[3];
  MonitorElement* h_dRPhoPFcand_Pho_unCleaned_[3];
  MonitorElement* h_SumPtOverPhoPt_ChHad_Cleaned_[3];
  MonitorElement* h_SumPtOverPhoPt_NeuHad_Cleaned_[3];
  MonitorElement* h_SumPtOverPhoPt_Pho_Cleaned_[3];
  MonitorElement* h_SumPtOverPhoPt_ChHad_unCleaned_[3];
  MonitorElement* h_SumPtOverPhoPt_NeuHad_unCleaned_[3];
  MonitorElement* h_SumPtOverPhoPt_Pho_unCleaned_[3];


  


};





#endif




