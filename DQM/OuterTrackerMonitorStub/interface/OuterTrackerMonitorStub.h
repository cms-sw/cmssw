#ifndef OuterTrackerMonitorStub_OuterTrackerMonitorStub_h
#define OuterTrackerMonitorStub_OuterTrackerMonitorStub_h

#include <vector>
#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include <DQMServices/Core/interface/DQMStore.h>
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
  MonitorElement* hStub_Barrel_XY = 0;  //TTStub barrel y vs x
  MonitorElement* hStub_Barrel_XY_Zoom = 0;  //TTStub barrel y vs x zoom
  MonitorElement* hStub_Endcap_Fw_XY = 0; //TTStub Forward Endcap y vs. x
  MonitorElement* hStub_Endcap_Bw_XY = 0; //TTStub Backward Endcap y vs. x
  MonitorElement* hStub_RZ = 0; // TTStub #rho vs. z
  MonitorElement* hStub_Endcap_Fw_RZ_Zoom = 0; // TTStub Forward Endcap #rho vs. z
  MonitorElement* hStub_Endcap_Bw_RZ_Zoom = 0; // TTStub Backward Endcap #rho vs. z
	

 private:
  DQMStore* dqmStore_;
  edm::ParameterSet conf_;

  std::string topFolderName_;
};
#endif


