#ifndef PhotonAnalyzer_H
#define PhotonAnalyzer_H
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
//
//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"
//
#include <map>
#include <vector>


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
      
  std::string fName_;
  DQMStore *dbe_;
  int verbosity_;

  int nEvt_;
  int nEntry_;
  int nMCPho_;
  int nMatched_;
  edm::ParameterSet parameters_;
      
           
  std::string photonCollectionProducer_;       
  std::string photonCollection_;

  


  // SC from reco photons
  MonitorElement* h_scEta_;
  MonitorElement* h_scPhi_;
  MonitorElement* h_scEtaPhi_;
 
  std::vector<MonitorElement*> h_scE_;
  std::vector<MonitorElement*> h_scEt_;

  std::vector<MonitorElement*> h_r9_;  
  std::vector<MonitorElement*> h_phoE_;
  std::vector<MonitorElement*> h_phoEt_;
  

  //
  MonitorElement* h_phoEta_;
  MonitorElement* h_phoPhi_;
  //
  


};

#endif
