#ifndef SiOuterTracker_OuterTrackerMonitorTTCluster_h
#define SiOuterTracker_OuterTrackerMonitorTTCluster_h

#include <vector>
#include <memory>
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"


class DQMStore;

class OuterTrackerMonitorTTCluster : public DQMEDAnalyzer {

public:
  explicit OuterTrackerMonitorTTCluster(const edm::ParameterSet&);
  ~OuterTrackerMonitorTTCluster() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

  // TTCluster stacks
  MonitorElement* Cluster_IMem_Barrel = nullptr;
  MonitorElement* Cluster_IMem_Endcap_Disc = nullptr;
  MonitorElement* Cluster_IMem_Endcap_Ring = nullptr;
  MonitorElement* Cluster_IMem_Endcap_Ring_Fw[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
  MonitorElement* Cluster_IMem_Endcap_Ring_Bw[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
  MonitorElement* Cluster_OMem_Barrel = nullptr;
  MonitorElement* Cluster_OMem_Endcap_Disc = nullptr;
  MonitorElement* Cluster_OMem_Endcap_Ring = nullptr;
  MonitorElement* Cluster_OMem_Endcap_Ring_Fw[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
  MonitorElement* Cluster_OMem_Endcap_Ring_Bw[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
  MonitorElement* Cluster_W = nullptr;
  MonitorElement* Cluster_Phi = nullptr;
  MonitorElement* Cluster_R = nullptr;
  MonitorElement* Cluster_Eta = nullptr;

  MonitorElement* Cluster_Barrel_XY = nullptr;
  MonitorElement* Cluster_Endcap_Fw_XY = nullptr;
  MonitorElement* Cluster_Endcap_Bw_XY = nullptr;
  MonitorElement* Cluster_RZ = nullptr;

 private:
  edm::ParameterSet conf_;
  edm::EDGetTokenT<edmNew::DetSetVector< TTCluster< Ref_Phase2TrackerDigi_ > > >  tagTTClustersToken_;
  std::string topFolderName_;
};
#endif
