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

#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/DetId/interface/DetId.h"

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
 **  $Date: 2012/06/27 11:54:07 $ 
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
  std::string photonProducer_;       
  std::string photonCollection_;
  std::string barrelRecHitProducer_;
  std::string barrelRecHitCollection_;
  std::string endcapRecHitProducer_;
  std::string endcapRecHitCollection_;
  std::string muonProducer_;       
  std::string muonCollection_;
  //
  std::string fName_;
  int verbosity_;
  edm::InputTag triggerEvent_;
  bool useTriggerFiltering_;
  bool splitHistosEBEE_;
  bool use2DHistos_;
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
  float mumuGammaInvMass(const reco::Muon & mu1,const reco::Muon & mu2, const reco::Photon& pho );
  bool  basicMuonSelection (  const reco::Muon & m );
  bool  muonSelection (  const reco::Muon & m,  const reco::BeamSpot& bs );
  bool  photonSelection (  const reco::Photon & p );


  ///photon histos
  MonitorElement* h1_mumuInvMass_;
  MonitorElement* h1_mumuGammaInvMass_;

  MonitorElement* h_phoE_;
  MonitorElement* h_phoEt_;

  MonitorElement* h_nPho_;

  MonitorElement* h_phoEta_;
  MonitorElement* h_phoPhi_;
  MonitorElement* h_scEta_;
  MonitorElement* h_scPhi_;

  MonitorElement* h_r9_;
  MonitorElement* h_r9VsEt_;
  MonitorElement* p_r9VsEt_;
  MonitorElement* h_r9VsEta_;
  MonitorElement* p_r9VsEta_;

  MonitorElement* h_e1x5VsEta_;
  MonitorElement* p_e1x5VsEta_;
  MonitorElement* h_e1x5VsEt_;
  MonitorElement* p_e1x5VsEt_;
        
  MonitorElement* h_e2x5VsEta_;
  MonitorElement* p_e2x5VsEta_;
  MonitorElement* h_e2x5VsEt_;
  MonitorElement* p_e2x5VsEt_;
  
  MonitorElement* h_r1x5VsEta_;
  MonitorElement* p_r1x5VsEta_;
  MonitorElement* h_r1x5VsEt_;
  MonitorElement* p_r1x5VsEt_;
        
  MonitorElement* h_r2x5VsEta_;
  MonitorElement* p_r2x5VsEta_;
  MonitorElement* h_r2x5VsEt_;
  MonitorElement* p_r2x5VsEt_;
        
  MonitorElement* h_maxEXtalOver3x3VsEta_;
  MonitorElement* p_maxEXtalOver3x3VsEta_;
  MonitorElement* h_maxEXtalOver3x3VsEt_;
  MonitorElement* p_maxEXtalOver3x3VsEt_;

  MonitorElement* h_phoSigmaIetaIeta_;
  MonitorElement* h_sigmaIetaIetaVsEta_; 
  MonitorElement* p_sigmaIetaIetaVsEta_;

  MonitorElement* h_nTrackIsolSolid_;     
  MonitorElement* h_nTrackIsolSolidVsEt_; 
  MonitorElement* p_nTrackIsolSolidVsEt_; 
  MonitorElement* h_nTrackIsolSolidVsEta_;
  MonitorElement* p_nTrackIsolSolidVsEta_;
  
  MonitorElement* h_nTrackIsolHollow_;    
  MonitorElement* h_nTrackIsolHollowVsEt_; 
  MonitorElement* p_nTrackIsolHollowVsEt_; 
  MonitorElement* h_nTrackIsolHollowVsEta_;
  MonitorElement* p_nTrackIsolHollowVsEta_;
  
  MonitorElement* h_trackPtSumSolid_;      
  MonitorElement* h_trackPtSumSolidVsEt_;  
  MonitorElement* p_trackPtSumSolidVsEt_;  
  MonitorElement* h_trackPtSumSolidVsEta_; 
  MonitorElement* p_trackPtSumSolidVsEta_; 
 
  MonitorElement* h_trackPtSumHollow_;     
  MonitorElement* h_trackPtSumHollowVsEt_; 
  MonitorElement* p_trackPtSumHollowVsEt_; 
  MonitorElement* h_trackPtSumHollowVsEta_; 
  MonitorElement* p_trackPtSumHollowVsEta_; 

  MonitorElement* h_ecalSum_;    
  MonitorElement* h_ecalSumVsEt_;  
  MonitorElement* p_ecalSumVsEt_;  
  MonitorElement* h_ecalSumVsEta_; 
  MonitorElement* p_ecalSumVsEta_; 
  
  MonitorElement* h_hcalSum_;      
  MonitorElement* h_hcalSumVsEt_;  
  MonitorElement* p_hcalSumVsEt_;  
  MonitorElement* h_hcalSumVsEta_; 
  MonitorElement* p_hcalSumVsEta_; 
  
  MonitorElement* h_hOverE_;       
  MonitorElement* p_hOverEVsEt_;   
  MonitorElement* p_hOverEVsEta_;  
  MonitorElement* h_h1OverE_;      
  MonitorElement* h_h2OverE_;      

  ///barrel only histos
  MonitorElement* h_phoEBarrel_;
  MonitorElement* h_phoEtBarrel_;

  MonitorElement* h_nPhoBarrel_;

  MonitorElement* h_phoEtaBarrel_;
  MonitorElement* h_phoPhiBarrel_;
  MonitorElement* h_scEtaBarrel_;
  MonitorElement* h_scPhiBarrel_;

  MonitorElement* h_r9Barrel_;
  MonitorElement* h_r9VsEtBarrel_;
  MonitorElement* p_r9VsEtBarrel_;
  MonitorElement* h_r9VsEtaBarrel_;
  MonitorElement* p_r9VsEtaBarrel_;

  MonitorElement* h_e1x5VsEtaBarrel_;
  MonitorElement* p_e1x5VsEtaBarrel_;
  MonitorElement* h_e1x5VsEtBarrel_;
  MonitorElement* p_e1x5VsEtBarrel_;
        
  MonitorElement* h_e2x5VsEtaBarrel_;
  MonitorElement* p_e2x5VsEtaBarrel_;
  MonitorElement* h_e2x5VsEtBarrel_;
  MonitorElement* p_e2x5VsEtBarrel_;
  
  MonitorElement* h_r1x5VsEtaBarrel_;
  MonitorElement* p_r1x5VsEtaBarrel_;
  MonitorElement* h_r1x5VsEtBarrel_;
  MonitorElement* p_r1x5VsEtBarrel_;
        
  MonitorElement* h_r2x5VsEtaBarrel_;
  MonitorElement* p_r2x5VsEtaBarrel_;
  MonitorElement* h_r2x5VsEtBarrel_;
  MonitorElement* p_r2x5VsEtBarrel_;
        
  MonitorElement* h_maxEXtalOver3x3VsEtaBarrel_;
  MonitorElement* p_maxEXtalOver3x3VsEtaBarrel_;
  MonitorElement* h_maxEXtalOver3x3VsEtBarrel_;
  MonitorElement* p_maxEXtalOver3x3VsEtBarrel_;

  MonitorElement* h_phoSigmaIetaIetaBarrel_;
  MonitorElement* h_sigmaIetaIetaVsEtaBarrel_; 
  MonitorElement* p_sigmaIetaIetaVsEtaBarrel_;

  MonitorElement* h_nTrackIsolSolidBarrel_;     
  MonitorElement* h_nTrackIsolSolidVsEtBarrel_; 
  MonitorElement* p_nTrackIsolSolidVsEtBarrel_; 
  MonitorElement* h_nTrackIsolSolidVsEtaBarrel_;
  MonitorElement* p_nTrackIsolSolidVsEtaBarrel_;
  
  MonitorElement* h_nTrackIsolHollowBarrel_;    
  MonitorElement* h_nTrackIsolHollowVsEtBarrel_; 
  MonitorElement* p_nTrackIsolHollowVsEtBarrel_; 
  MonitorElement* h_nTrackIsolHollowVsEtaBarrel_;
  MonitorElement* p_nTrackIsolHollowVsEtaBarrel_;
  
  MonitorElement* h_trackPtSumSolidBarrel_;      
  MonitorElement* h_trackPtSumSolidVsEtBarrel_;  
  MonitorElement* p_trackPtSumSolidVsEtBarrel_;  
  MonitorElement* h_trackPtSumSolidVsEtaBarrel_; 
  MonitorElement* p_trackPtSumSolidVsEtaBarrel_; 
 
  MonitorElement* h_trackPtSumHollowBarrel_;     
  MonitorElement* h_trackPtSumHollowVsEtBarrel_; 
  MonitorElement* p_trackPtSumHollowVsEtBarrel_; 
  MonitorElement* h_trackPtSumHollowVsEtaBarrel_; 
  MonitorElement* p_trackPtSumHollowVsEtaBarrel_; 

  MonitorElement* h_ecalSumBarrel_;    
  MonitorElement* h_ecalSumVsEtBarrel_;  
  MonitorElement* p_ecalSumVsEtBarrel_;  
  MonitorElement* h_ecalSumVsEtaBarrel_; 
  MonitorElement* p_ecalSumVsEtaBarrel_; 
  
  MonitorElement* h_hcalSumBarrel_;      
  MonitorElement* h_hcalSumVsEtBarrel_;  
  MonitorElement* p_hcalSumVsEtBarrel_;  
  MonitorElement* h_hcalSumVsEtaBarrel_; 
  MonitorElement* p_hcalSumVsEtaBarrel_; 
  
  MonitorElement* h_hOverEBarrel_;       
  MonitorElement* p_hOverEVsEtBarrel_;   
  MonitorElement* p_hOverEVsEtaBarrel_;  
  MonitorElement* h_h1OverEBarrel_;      
  MonitorElement* h_h2OverEBarrel_;      


  ///endcap only histos
  MonitorElement* h_phoEEndcap_;
  MonitorElement* h_phoEtEndcap_;

  MonitorElement* h_nPhoEndcap_;

  MonitorElement* h_phoEtaEndcap_;
  MonitorElement* h_phoPhiEndcap_;
  MonitorElement* h_scEtaEndcap_;
  MonitorElement* h_scPhiEndcap_;

  MonitorElement* h_r9Endcap_;
  MonitorElement* h_r9VsEtEndcap_;
  MonitorElement* p_r9VsEtEndcap_;
  MonitorElement* h_r9VsEtaEndcap_;
  MonitorElement* p_r9VsEtaEndcap_;

  MonitorElement* h_e1x5VsEtaEndcap_;
  MonitorElement* p_e1x5VsEtaEndcap_;
  MonitorElement* h_e1x5VsEtEndcap_;
  MonitorElement* p_e1x5VsEtEndcap_;
        
  MonitorElement* h_e2x5VsEtaEndcap_;
  MonitorElement* p_e2x5VsEtaEndcap_;
  MonitorElement* h_e2x5VsEtEndcap_;
  MonitorElement* p_e2x5VsEtEndcap_;
  
  MonitorElement* h_r1x5VsEtaEndcap_;
  MonitorElement* p_r1x5VsEtaEndcap_;
  MonitorElement* h_r1x5VsEtEndcap_;
  MonitorElement* p_r1x5VsEtEndcap_;
        
  MonitorElement* h_r2x5VsEtaEndcap_;
  MonitorElement* p_r2x5VsEtaEndcap_;
  MonitorElement* h_r2x5VsEtEndcap_;
  MonitorElement* p_r2x5VsEtEndcap_;
        
  MonitorElement* h_maxEXtalOver3x3VsEtaEndcap_;
  MonitorElement* p_maxEXtalOver3x3VsEtaEndcap_;
  MonitorElement* h_maxEXtalOver3x3VsEtEndcap_;
  MonitorElement* p_maxEXtalOver3x3VsEtEndcap_;

  MonitorElement* h_phoSigmaIetaIetaEndcap_;
  MonitorElement* h_sigmaIetaIetaVsEtaEndcap_; 
  MonitorElement* p_sigmaIetaIetaVsEtaEndcap_;

  MonitorElement* h_nTrackIsolSolidEndcap_;     
  MonitorElement* h_nTrackIsolSolidVsEtEndcap_; 
  MonitorElement* p_nTrackIsolSolidVsEtEndcap_; 
  MonitorElement* h_nTrackIsolSolidVsEtaEndcap_;
  MonitorElement* p_nTrackIsolSolidVsEtaEndcap_;
  
  MonitorElement* h_nTrackIsolHollowEndcap_;    
  MonitorElement* h_nTrackIsolHollowVsEtEndcap_; 
  MonitorElement* p_nTrackIsolHollowVsEtEndcap_; 
  MonitorElement* h_nTrackIsolHollowVsEtaEndcap_;
  MonitorElement* p_nTrackIsolHollowVsEtaEndcap_;
  
  MonitorElement* h_trackPtSumSolidEndcap_;      
  MonitorElement* h_trackPtSumSolidVsEtEndcap_;  
  MonitorElement* p_trackPtSumSolidVsEtEndcap_;  
  MonitorElement* h_trackPtSumSolidVsEtaEndcap_; 
  MonitorElement* p_trackPtSumSolidVsEtaEndcap_; 
 
  MonitorElement* h_trackPtSumHollowEndcap_;     
  MonitorElement* h_trackPtSumHollowVsEtEndcap_; 
  MonitorElement* p_trackPtSumHollowVsEtEndcap_; 
  MonitorElement* h_trackPtSumHollowVsEtaEndcap_; 
  MonitorElement* p_trackPtSumHollowVsEtaEndcap_; 

  MonitorElement* h_ecalSumEndcap_;    
  MonitorElement* h_ecalSumVsEtEndcap_;  
  MonitorElement* p_ecalSumVsEtEndcap_;  
  MonitorElement* h_ecalSumVsEtaEndcap_; 
  MonitorElement* p_ecalSumVsEtaEndcap_; 
  
  MonitorElement* h_hcalSumEndcap_;      
  MonitorElement* h_hcalSumVsEtEndcap_;  
  MonitorElement* p_hcalSumVsEtEndcap_;  
  MonitorElement* h_hcalSumVsEtaEndcap_; 
  MonitorElement* p_hcalSumVsEtaEndcap_; 
  
  MonitorElement* h_hOverEEndcap_;       
  MonitorElement* p_hOverEVsEtEndcap_;   
  MonitorElement* p_hOverEVsEtaEndcap_;  
  MonitorElement* h_h1OverEEndcap_;      
  MonitorElement* h_h2OverEEndcap_;      


};





#endif




