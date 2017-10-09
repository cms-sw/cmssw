#ifndef Phase2OuterTracker_OuterTrackerMonitorTTClusterClient_h
#define Phase2OuterTracker_OuterTrackerMonitorTTClusterClient_h


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

class DQMStore;

class OuterTrackerMonitorTTClusterClient : public edm::EDAnalyzer {

public:
  explicit OuterTrackerMonitorTTClusterClient(const edm::ParameterSet&);
  ~OuterTrackerMonitorTTClusterClient();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  //virtual void beginJob() ;
  virtual void endJob() ;
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
  
};
#endif
