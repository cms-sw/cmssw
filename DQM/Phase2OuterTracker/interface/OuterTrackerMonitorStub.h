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
  MonitorElement* Stub_Barrel = 0; //TTStub per layer
  MonitorElement* Stub_Endcap_Disc = 0; // TTStubs per disc
  MonitorElement* Stub_Endcap_Disc_Fw = 0; //TTStub per disc	
  MonitorElement* Stub_Endcap_Disc_Bw = 0; //TTStub per disc
  MonitorElement* Stub_Endcap_Ring = 0; // TTStubs per ring
  MonitorElement* Stub_Endcap_Ring_Fw[5] = {0, 0, 0, 0, 0}; // TTStubs per EC ring
  MonitorElement* Stub_Endcap_Ring_Bw[5] = {0, 0, 0, 0, 0}; //TTStub per EC ring
  
  // * Stub Eta distribution * //
  MonitorElement* Stub_Eta = 0; //TTstub eta distribution
  
  // * STUB Displacement - offset * //
  MonitorElement* Stub_Barrel_W = 0; //TTstub Pos-Corr Displacement (layer)
  MonitorElement* Stub_Barrel_O = 0; // TTStub Offset (layer)
  MonitorElement* Stub_Endcap_Disc_W = 0; // TTstub Pos-Corr Displacement (disc)
  MonitorElement* Stub_Endcap_Disc_O = 0; // TTStub Offset (disc)
  MonitorElement* Stub_Endcap_Ring_W = 0; // TTstub Pos-Corr Displacement (EC ring)
  MonitorElement* Stub_Endcap_Ring_O = 0; // TTStub Offset (EC ring)
  MonitorElement* Stub_Endcap_Ring_W_Fw[5] = {0, 0, 0, 0, 0}; // TTstub Pos-Corr Displacement (EC ring)
  MonitorElement* Stub_Endcap_Ring_O_Fw[5] = {0, 0, 0, 0, 0}; // TTStub Offset (EC ring)
  MonitorElement* Stub_Endcap_Ring_W_Bw[5] = {0, 0, 0, 0, 0}; // TTstub Pos-Corr Displacement (EC ring)
  MonitorElement* Stub_Endcap_Ring_O_Bw[5] = {0, 0, 0, 0, 0}; // TTStub Offset (EC ring)

 private:
  DQMStore* dqmStore_;
  edm::ParameterSet conf_;
  edm::InputTag tagTTStubs_;

  std::string topFolderName_;
};
#endif


