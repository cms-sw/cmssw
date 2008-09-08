#ifndef PhotonAnalyzer_H
#define PhotonAnalyzer_H
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

//
#include <map>
#include <vector>
/** \class PhotonAnalyzer
 **  
 **
 **  $Id: PhotonAnalyzer
 **  $Date: 2008/09/05 15:37:31 $ 
 **  authors: 
 **   Nancy Marinelli, U. of Notre Dame, US  
 **   Jamie Antonelli, U. of Notre Dame, US
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
  void doProfileX(TH2 * th2, MonitorElement* me);
  void doProfileX(MonitorElement * th2m, MonitorElement* me);

      
  std::string fName_;
  DQMStore *dbe_;
  int verbosity_;

  int nEvt_;
  int nEntry_;
  int nMCPho_;
  int nMatched_;
  edm::ParameterSet parameters_;
  edm::ESHandle<CaloGeometry> theCaloGeom_;	    
  edm::ESHandle<CaloTopology> theCaloTopo_;

           
  std::string photonProducer_;       
  std::string photonCollection_;
 

  
  double minPhoEtCut_;


  double lip_;

  double ecalEtaStrip_;



  double cutStep_;
  int numberOfSteps_;
  
  std::vector<MonitorElement*> h_nTrackIsolSolid_;
  std::vector<MonitorElement*> h_trackPtSumSolid_;
  std::vector<MonitorElement*> h_ecalSum_;
  std::vector<MonitorElement*> h_hcalSum_;

  std::vector<MonitorElement*> p_nTrackIsolSolid_;
  std::vector<MonitorElement*> p_trackPtSumSolid_;
  std::vector<MonitorElement*> p_ecalSum_;
  std::vector<MonitorElement*> p_hcalSum_;

  std::vector<MonitorElement*> h_nTrackIsolHollow_;
  std::vector<MonitorElement*> h_trackPtSumHollow_;
  std::vector<MonitorElement*> p_nTrackIsolHollow_;
  std::vector<MonitorElement*> p_trackPtSumHollow_;

  std::vector<MonitorElement*> h_phoE_part_;
  std::vector<std::vector<MonitorElement*> > h_phoE_isol_;
  std::vector<std::vector<std::vector<MonitorElement*> > > h_phoE_;

  std::vector<MonitorElement*> h_phoEt_part_;
  std::vector<std::vector<MonitorElement*> > h_phoEt_isol_;
  std::vector<std::vector<std::vector<MonitorElement*> > > h_phoEt_;

  std::vector<MonitorElement*> h_r9_part_;
  std::vector<std::vector<MonitorElement*> > h_r9_isol_;
  std::vector<std::vector<std::vector<MonitorElement*> > > h_r9_;

  std::vector<MonitorElement*> h_nPho_part_;
  std::vector<std::vector<MonitorElement*> > h_nPho_isol_;
  std::vector<std::vector<std::vector<MonitorElement*> > > h_nPho_;

  std::vector<MonitorElement*> h_phoDistribution_part_;
  std::vector<std::vector<MonitorElement*> > h_phoDistribution_isol_;
  std::vector<std::vector<std::vector<MonitorElement*> > > h_phoDistribution_;

  std::vector<MonitorElement*> h_phoEta_isol_;
  std::vector<std::vector<MonitorElement*> > h_phoEta_;
  std::vector<MonitorElement*> h_phoPhi_isol_;
  std::vector<std::vector<MonitorElement*> > h_phoPhi_;

  std::vector<MonitorElement*> p_r9VsEt_isol_;
  std::vector<std::vector<MonitorElement*> > p_r9VsEt_;

  //
  //
  


};

#endif
