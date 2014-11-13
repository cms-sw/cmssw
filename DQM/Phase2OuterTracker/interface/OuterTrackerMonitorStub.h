#ifndef Phase2OuterTracker_OuterTrackerMonitorStub_h
#define Phase2OuterTracker_OuterTrackerMonitorStub_h

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
#include "DQM/SiStripCommon/interface/TkHistoMap.h"

class DQMStore;

class OuterTrackerMonitorStub : public edm::EDAnalyzer {

public:
  explicit OuterTrackerMonitorStub(const edm::ParameterSet&);
  ~OuterTrackerMonitorStub();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
 
  // TTStub stacks
  // * Global position of the stubs * //
  MonitorElement* Stub_Barrel_XY = 0;  //TTStub barrel y vs x
  MonitorElement* Stub_Barrel_XY_Zoom = 0;  //TTStub barrel y vs x zoom
  MonitorElement* Stub_Endcap_Fw_XY = 0; //TTStub Forward Endcap y vs. x
  MonitorElement* Stub_Endcap_Bw_XY = 0; //TTStub Backward Endcap y vs. x
  MonitorElement* Stub_RZ = 0; // TTStub #rho vs. z
  MonitorElement* Stub_Endcap_Fw_RZ_Zoom = 0; // TTStub Forward Endcap #rho vs. z
  MonitorElement* Stub_Endcap_Bw_RZ_Zoom = 0; // TTStub Backward Endcap #rho vs. z
  
  // * Number of stubs * //
  MonitorElement* Stub_Endcap = 0; // TTStubs stack
  MonitorElement* Stub_Barrel = 0; //TTStub stack
  MonitorElement* Stub_Endcap_Fw = 0; //TTStub stack	
  MonitorElement* Stub_Endcap_Bw = 0; //TTStub stack	
  
  // * Stub Eta distribution * //
  MonitorElement* Stub_Eta = 0; //TTstub eta distribution
  
  // * STUB Displacement - offset * //
  MonitorElement* Stub_Barrel_W = 0; //TTstub Pos-Corr Displacement (layer)
  MonitorElement* Stub_Barrel_O = 0; // TTStub Offset (layer)
  MonitorElement* Stub_Endcap_W = 0; // TTstub Pos-Corr Displacement (layer)
  MonitorElement* Stub_Endcap_O = 0; // TTStub Offset (layer)

 private:
  DQMStore* dqmStore_;
  edm::ParameterSet conf_;

  std::string topFolderName_;
};
#endif


