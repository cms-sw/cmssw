#ifndef Phase2OuterTracker_OuterTrackerMonitorCluster_h
#define Phase2OuterTracker_OuterTrackerMonitorCluster_h

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

class OuterTrackerMonitorCluster : public edm::EDAnalyzer {

public:
  explicit OuterTrackerMonitorCluster(const edm::ParameterSet&);
  ~OuterTrackerMonitorCluster();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  virtual void beginRun(const edm::Run&, const edm::EventSetup&);
 
	// TTCluster stacks
	MonitorElement* Cluster_IMem_Barrel = 0;
	MonitorElement* Cluster_IMem_Endcap_Disc = 0;
  MonitorElement* Cluster_IMem_Endcap_Ring = 0;
  MonitorElement* Cluster_IMem_Endcap_Ring_Fw[5] = {0, 0, 0, 0, 0};
  MonitorElement* Cluster_IMem_Endcap_Ring_Bw[5] = {0, 0, 0, 0, 0};
	MonitorElement* Cluster_OMem_Barrel = 0;
	MonitorElement* Cluster_OMem_Endcap_Disc = 0;
  MonitorElement* Cluster_OMem_Endcap_Ring = 0;
  MonitorElement* Cluster_OMem_Endcap_Ring_Fw[5] = {0, 0, 0, 0, 0};
  MonitorElement* Cluster_OMem_Endcap_Ring_Bw[5] = {0, 0, 0, 0, 0};
	MonitorElement* Cluster_W = 0;
  MonitorElement* Cluster_Eta = 0;
	
	MonitorElement* Cluster_Barrel_XY = 0;
  MonitorElement* Cluster_Barrel_XY_Zoom = 0;
  MonitorElement* Cluster_Endcap_Fw_XY = 0;
  MonitorElement* Cluster_Endcap_Bw_XY = 0;
  MonitorElement* Cluster_RZ = 0;
  MonitorElement* Cluster_Endcap_Fw_RZ_Zoom = 0;
  MonitorElement* Cluster_Endcap_Bw_RZ_Zoom = 0;

 private:
  DQMStore* dqmStore_;
  edm::ParameterSet conf_;
  edm::InputTag tagTTClusters_;

  std::string topFolderName_;
};
#endif
