#ifndef OuterTrackerMonitorCluster_OuterTrackerMonitorCluster_h
#define OuterTrackerMonitorCluster_OuterTrackerMonitorCluster_h

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

#include <vector>

class DQMStore;
class SiStripDetCabling;
class SiStripCluster;
class SiStripDCSStatus;
class GenericTriggerEventFlag;

class OuterTrackerMonitorCluster : public edm::EDAnalyzer {

public:
  explicit OuterTrackerMonitorCluster(const edm::ParameterSet&);
  ~OuterTrackerMonitorCluster();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  //virtual void beginJob() ;
  virtual void endJob() ;
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
 
  MonitorElement* NumberOfStripClus = 0;
	
	

 private:
  DQMStore* dqmStore_;
  edm::ParameterSet conf_;
  edm::InputTag clusterProducerStrip_;

  std::string topFolderName_;
  
};
#endif
