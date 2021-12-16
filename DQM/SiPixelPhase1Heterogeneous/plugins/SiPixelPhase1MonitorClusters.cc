// -*- C++ -*-
///bookLayer
// Package:    SiPixelPhase1MonitorClusters
// Class:      SiPixelPhase1MonitorClusters
//
/**\class SiPixelPhase1MonitorClusters SiPixelPhase1MonitorClusters.cc 
*/
//
// Author: Suvankar Roy Chowdhury
//
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
// Geometry
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
// DQM Histograming
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/SiPixelDigi/interface/SiPixelDigisSoA.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

class SiPixelPhase1MonitorClusters : public DQMEDAnalyzer {
public:
  explicit SiPixelPhase1MonitorClusters(const edm::ParameterSet&);
  ~SiPixelPhase1MonitorClusters() override = default;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::EDGetTokenT<SiPixelClusterCollectionNew> tokenCluster_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopoToken_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeomToken_;
  std::string topFolderName_;
  MonitorElement* hnClusters;
  const TrackerTopology* trackerTopology_;
	
  MonitorElement* hnClusters;
  MonitorElement* hClustersCharge;
  MonitorElement* hClustersSize;
  
  MonitorElement* hnClustersBarrel;
  MonitorElement* hClustersBarrelCharge;
  MonitorElement* hClustersBarrelSize;
  
  MonitorElement* hnClustersEndcap;
  MonitorElement* hClustersEndcapCharge;
  MonitorElement* hClustersEndcapSize;

  MonitorElement* hnClustersBarrelLayer1;
  MonitorElement* hnClustersBarrelLayer2;
  MonitorElement* hnClustersBarrelLayer3;
  MonitorElement* hnClustersBarrelLayer4;

  MonitorElement* hnClustersBarrelLayer1Charge;
  MonitorElement* hnClustersBarrelLayer2Charge;
  MonitorElement* hnClustersBarrelLayer3Charge;
  MonitorElement* hnClustersBarrelLayer4Charge;
	
  MonitorElement* hnClustersBarrelLayer1Charge;
  MonitorElement* hnClustersBarrelLayer2Charge;
  MonitorElement* hnClustersBarrelLayer3Charge;
  MonitorElement* hnClustersBarrelLayer4Charge;

  MonitorElement* hnClustersEndcapDiskm1;
  MonitorElement* hnClustersEndcapDiskm2;
  MonitorElement* hnClustersEndcapDiskm3;
  MonitorElement* hnClustersEndcapDiskp1;
  MonitorElement* hnClustersEndcapDiskp2;
  MonitorElement* hnClustersEndcapDiskp3;

  MonitorElement* hDigisEndcapDiskm1Charge;
  MonitorElement* hDigisEndcapDiskm2Charge;
  MonitorElement* hDigisEndcapDiskm3Charge;
  MonitorElement* hDigisEndcapDiskp1Charge;
  MonitorElement* hDigisEndcapDiskp2Charge;
  MonitorElement* hDigisEndcapDiskp3Charge;
	
  MonitorElement* hClustersEndcapDiskm1Size;
  MonitorElement* hClustersEndcapDiskm2Size;
  MonitorElement* hClustersEndcapDiskm3Size;
  MonitorElement* hClustersEndcapDiskp1Size;
  MonitorElement* hClustersEndcapDiskp2Size;
  MonitorElement* hClustersEndcapDiskp3Size;
};

//
// constructors
//

SiPixelPhase1MonitorClusters::SiPixelPhase1MonitorClusters(const edm::ParameterSet& iConfig) {
  tokenCluster_ = consumes<SiPixelClusterCollectionNew>(iConfig.getParameter<edm::InputTag>("pixelClusters"));
  topFolderName_ = iConfig.getParameter<std::string>("TopFolderName");  //"SiPixelHeterogeneous/PixelTrackSoA";
  trackerGeomToken_ = esConsumes<TrackerGeometry, TrackerDigiGeometryRecord>();
  trackerTopoToken_ = esConsumes<TrackerTopology, TrackerTopologyRcd>();
}

//
// -- Analyze
//
void SiPixelPhase1MonitorClusters::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // get geometry
  edm::ESHandle<TrackerGeometry> tracker = iSetup.getHandle(trackerGeomToken_);
  assert(tracker.isValid());
  // TrackerTopology for module informations
  edm::ESHandle<TrackerTopology> trackerTopologyHandle = iSetup.getHandle(trackerTopoToken_);
  trackerTopology_ = trackerTopologyHandle.product();

  edm::Handle<SiPixelClusterCollectionNew> input;
  iEvent.getByToken(tokenCluster_, input);
  if (!input.isValid()){
    edm::LogWarning("SiPixelPhase1MonitorClusters") << "No Valid siPixelClustersPreSplitting found returning!" << std::endl;
  }
  else{
    SiPixelClusterCollectionNew::const_iterator it;
    uint32_t nClus = 0;
    for (it = input->begin(); it != input->end(); ++it) {
      nClus += it->size();
      /* DetId id = it->detId();
      uint32_t subdetid = (id.subdetId());
      if (subdetid == PixelSubdetector::PixelBarrel) {
	edm::LogWarning("SiPixelPhase1MonitorClusters") << "PX Barrel:  DetId " <<id.rawId()<<" Layer "<<trackerTopology_->pxbLayer(id)<<std::endl;
      }
      if (subdetid == PixelSubdetector::PixelEndcap) {
	edm::LogWarning("SiPixelPhase1MonitorClusters") << "PX Endcaps:  DetId " <<id.rawId()<<" Side "<<trackerTopology_->pxfSide(id)<<" Disk "<<trackerTopology_->pxfDisk(id)<<std::endl;
	}*/
    }
    edm::LogWarning("SiPixelPhase1MonitorClusters") << "Found "<<nClus<<" Clusters!" << std::endl;
    hnClusters->Fill(nClus);
  }
}

//
// -- Book Histograms
//
void SiPixelPhase1MonitorClusters::bookHistograms(DQMStore::IBooker& ibooker,
                                                  edm::Run const& iRun,
                                                  edm::EventSetup const& iSetup) {
  ibooker.cd();
  ibooker.setCurrentFolder(topFolderName_);
  hnClusters = ibooker.book1D("nClusters", ";Number of Clusters per event;#entries", 1001, -0.5, 10000.5);
  hnClusters2 = ibooker.book1D("nClusters2", ";Number of Clusters per event;#entries", 1001, -0.5, 10000.5);
}

void SiPixelPhase1MonitorClusters::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // monitorpixelTrackSoA
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pixelClusters", edm::InputTag("siPixelClustersPreSplitting"));
  desc.add<std::string>("TopFolderName", "SiPixelHeterogeneous/PixelClusters");
  descriptions.addWithDefaultLabel(desc);
}
DEFINE_FWK_MODULE(SiPixelPhase1MonitorClusters);
