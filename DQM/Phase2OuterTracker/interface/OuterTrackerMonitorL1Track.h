#ifndef Phase2OuterTracker_OuterTrackerMonitorL1Track_h
#define Phase2OuterTracker_OuterTrackerMonitorL1Track_h

#include <vector>
#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/DetSetVector.h"

class DQMStore;

class OuterTrackerMonitorL1Track : public edm::EDAnalyzer {

public:
  explicit OuterTrackerMonitorL1Track(const edm::ParameterSet&);
  ~OuterTrackerMonitorL1Track();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
 
  MonitorElement* L1Track_3Stubs_NTracks = 0;
  
  MonitorElement* L1Track_3Stubs_Pt = 0; 
  MonitorElement* L1Track_3Stubs_Phi = 0; 
  MonitorElement* L1Track_3Stubs_Eta = 0;
  MonitorElement* L1Track_3Stubs_VtxZ0 = 0; 
  MonitorElement* L1Track_3Stubs_Chi2 = 0; 
  MonitorElement* L1Track_3Stubs_Chi2R = 0;
  
  MonitorElement* L1Track_3Stubs_Chi2_NStubs = 0;
  MonitorElement* L1Track_3Stubs_Chi2R_NStubs = 0;
  
  MonitorElement* L1Track_2Stubs_NTracks = 0;
  
  MonitorElement* L1Track_2Stubs_Pt = 0; 
  MonitorElement* L1Track_2Stubs_Phi = 0; 
  MonitorElement* L1Track_2Stubs_Eta = 0; 
  MonitorElement* L1Track_2Stubs_Theta = 0; 
  MonitorElement* L1Track_2Stubs_VtxZ0 = 0; 
  MonitorElement* L1Track_2Stubs_Chi2 = 0; 
  MonitorElement* L1Track_2Stubs_Chi2R = 0; 
  
  MonitorElement* L1Track_2Stubs_Chi2_NStubs = 0;
  MonitorElement* L1Track_2Stubs_Chi2R_NStubs = 0;
  
  MonitorElement* L1Track_NStubs_PhiSector = 0; 
  MonitorElement* L1Track_NStubs_EtaWedge = 0; 
  MonitorElement* L1Track_PhiSector_L1Track_Phi = 0; 
  MonitorElement* L1Track_EtaWedge_L1Track_Eta = 0; 
  MonitorElement* L1Track_NStubs = 0; 
  MonitorElement* L1Track_NTracks = 0; 


 private:
  DQMStore* dqmStore_;
  edm::ParameterSet conf_;

  std::string topFolderName_;
};
#endif
