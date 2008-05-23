#ifndef PhotonAnalyzer_H
#define PhotonAnalyzer_H
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
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

//
#include <map>
#include <vector>
/** \class PhotonAnalyzer
 **  
 **
 **  $Id: PhotonAnalyzer
 **  $Date: $ 
 **  $Revision: $
 **  \author Nancy Marinelli, U. of Notre Dame, US
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


class PhotonAnalyzer : public edm::EDAnalyzer
{

 public:
   
  //
  explicit PhotonAnalyzer( const edm::ParameterSet& ) ;
  virtual ~PhotonAnalyzer();
                                   
      
  virtual void analyze( const edm::Event&, const edm::EventSetup& ) ;
  virtual void beginJob( const edm::EventSetup& ) ;
  virtual void endJob() ;

 private:
  //

  float  phiNormalization( float& a);
      
  std::string fName_;
  DQMStore *dbe_;
  int verbosity_;

  int nEvt_;
  int nEntry_;
  int nMCPho_;
  int nMatched_;
  edm::ParameterSet parameters_;
  edm::ESHandle<CaloGeometry> theCaloGeom_;	    
           
  std::string photonCollectionProducer_;       
  std::string photonCollection_;
  std::string  bcProducer_;
  std::string  bcBarrelCollection_;
  std::string  bcEndcapCollection_;
  std::string hbheLabel_;
  std::string hbheInstanceName_;
 

  edm::InputTag scBarrelProducer_;
  edm::InputTag scEndcapProducer_;
 
  edm::InputTag barrelEcalHits_;
  edm::InputTag endcapEcalHits_;


  edm::InputTag tracksInputTag_;
  
  double minPhoEtCut_;
  double trkIsolExtRadius_;
  double trkIsolInnRadius_;
  double trkPtLow_;
  double lip_;
  double ecalIsolRadius_;
  double bcEtLow_;
  double hcalIsolExtRadius_;
  double hcalIsolInnRadius_;
  double hcalHitEtLow_;
  int  numOfTracksInCone_;
  double trkPtSumCut_;
  double ecalEtSumCut_;
  double hcalEtSumCut_;

  MonitorElement* p_nTrackIsol_;
  MonitorElement* p_trackPtSum_;
  MonitorElement* p_ecalSum_;
  MonitorElement* p_hcalSum_;
  

  MonitorElement* h_nPho_[2][3];
  MonitorElement* h_scEta_[2];
  MonitorElement* h_scPhi_[2];
  MonitorElement* h_scEtaPhi_[2];
 
  MonitorElement* h_scE_[2][3];
  MonitorElement* h_scEt_[2][3];

  MonitorElement* h_r9_[2][3];  
  MonitorElement* h_phoE_[2][3];
  MonitorElement* h_phoEt_[2][3];
  MonitorElement* h_phoEta_[2];
  MonitorElement* h_phoPhi_[2];

  /// conversion infos
  MonitorElement* h_nConv_[2][3];
  MonitorElement* h_convEta_[2];
  MonitorElement* h_convPhi_[2];
  MonitorElement* h_r9VsNofTracks_[2][3];
  MonitorElement* h_EoverPTracks_[2][3];
  MonitorElement* p_tk_nHitsVsEta_[2]; 
  MonitorElement* h_tkChi2_[2];
  MonitorElement* h_DPhiTracksAtVtx_[2][3];
  MonitorElement* h_DCotTracks_[2][3];
  MonitorElement* h_invMass_[2][3];
  MonitorElement* h_DPhiBCTrackAtEcal_[2][3];
  MonitorElement* h_DEtaBCTrackAtEcal_[2][3];

  MonitorElement* h_convVtxRvsZ_[2]; 
  MonitorElement* h_zPVFromTracks_[2]; 

  //
  //
  


};

#endif
