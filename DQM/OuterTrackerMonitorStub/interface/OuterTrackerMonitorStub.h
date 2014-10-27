#ifndef OuterTrackerMonitorCluster_OuterTrackerMonitorCluster_h
#define OuterTrackerMonitorCluster_OuterTrackerMonitorCluster_h

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
  MonitorElement* hStub_Barrel_XY = 0;  //TTStub barrel y vs x
	

 private:
  DQMStore* dqmStore_;
  edm::ParameterSet conf_;

  std::string topFolderName_;
};
#endif


