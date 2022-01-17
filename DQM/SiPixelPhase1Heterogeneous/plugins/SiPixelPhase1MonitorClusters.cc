// -*- C++ -*-
///bookLayer
// Package:    SiPixelPhase1MonitorClusters
// Class:      SiPixelPhase1MonitorClusters
//
/**\class SiPixelPhase1MonitorClusters SiPixelPhase1MonitorClusters.cc
*/
//
// Author: Tommaso Tedeschi
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

  MonitorElement* hClustersBarrelLayer1Charge;
  MonitorElement* hClustersBarrelLayer2Charge;
  MonitorElement* hClustersBarrelLayer3Charge;
  MonitorElement* hClustersBarrelLayer4Charge;

  MonitorElement* hClustersBarrelLayer1Size;
  MonitorElement* hClustersBarrelLayer2Size;
  MonitorElement* hClustersBarrelLayer3Size;
  MonitorElement* hClustersBarrelLayer4Size;

  MonitorElement* hnClustersEndcapDiskm1;
  MonitorElement* hnClustersEndcapDiskm2;
  MonitorElement* hnClustersEndcapDiskm3;
  MonitorElement* hnClustersEndcapDiskp1;
  MonitorElement* hnClustersEndcapDiskp2;
  MonitorElement* hnClustersEndcapDiskp3;

  MonitorElement* hClustersEndcapDiskm1Charge;
  MonitorElement* hClustersEndcapDiskm2Charge;
  MonitorElement* hClustersEndcapDiskm3Charge;
  MonitorElement* hClustersEndcapDiskp1Charge;
  MonitorElement* hClustersEndcapDiskp2Charge;
  MonitorElement* hClustersEndcapDiskp3Charge;

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
    edm::LogWarning("SiPixelPhase1MonitorClusters") << "Ciao" << std::endl;
    SiPixelClusterCollectionNew::const_iterator it;
    uint32_t nClusters = 0;
    uint32_t nClustersBarrel = 0;
    uint32_t nClustersBarrelLayer1 = 0;
    uint32_t nClustersBarrelLayer2 = 0;
    uint32_t nClustersBarrelLayer3 = 0;
    uint32_t nClustersBarrelLayer4 = 0;
    uint32_t nClustersEndcap = 0;
    uint32_t nClustersEndcapDiskm1 = 0;
    uint32_t nClustersEndcapDiskm2 = 0;
    uint32_t nClustersEndcapDiskm3 = 0;
    uint32_t nClustersEndcapDiskp1 = 0;
    uint32_t nClustersEndcapDiskp2 = 0;
    uint32_t nClustersEndcapDiskp3 = 0;
    for (it = input->begin(); it != input->end(); ++it) {
      const uint32_t nClustersEv = it->size();
      nClusters += nClustersEv;
      DetId id = it->detId();
      uint32_t subdetid = (id.subdetId());
        if (subdetid == PixelSubdetector::PixelBarrel) {
            edm::LogWarning("SiPixelPhase1MonitorClusters") << " PX Barrel:  DetId " <<id.rawId()<<" Layer "<<trackerTopology_->pxbLayer(id)<<std::endl;
            nClustersBarrel += nClustersEv;
            const uint32_t nLayer = trackerTopology_->pxbLayer(id);
            if(nLayer == 1){
                nClustersBarrelLayer1 += nClustersEv;
                for (SiPixelCluster const& cluster : *it){
                    int clustersize = cluster.size();
                    hClustersSize->Fill(clustersize);
                    hClustersBarrelSize->Fill(clustersize);
                    hClustersBarrelLayer1Size->Fill(clustersize);
                    int clustercharge = cluster.charge();
                    hClustersCharge->Fill(clustercharge);
                    hClustersBarrelCharge->Fill(clustercharge);
                    hClustersBarrelLayer1Charge->Fill(clustercharge);
                }
            }
            else if(nLayer == 2){
                nClustersBarrelLayer2 += nClustersEv;
                for (SiPixelCluster const& cluster : *it){
                    int clustersize = cluster.size();
                    hClustersSize->Fill(clustersize);
                    hClustersBarrelSize->Fill(clustersize);
                    hClustersBarrelLayer2Size->Fill(clustersize);
                    int clustercharge = cluster.charge();
                    hClustersCharge->Fill(clustercharge);
                    hClustersBarrelCharge->Fill(clustercharge);
                    hClustersBarrelLayer2Charge->Fill(clustercharge);
                }
            }

            else if(nLayer == 3){
                nClustersBarrelLayer3 += nClustersEv;
                for (SiPixelCluster const& cluster : *it){
                    int clustersize = cluster.size();
                    hClustersSize->Fill(clustersize);
                    hClustersBarrelSize->Fill(clustersize);
                    hClustersBarrelLayer3Size->Fill(clustersize);
                    int clustercharge = cluster.charge();
                    hClustersCharge->Fill(clustercharge);
                    hClustersBarrelCharge->Fill(clustercharge);
                    hClustersBarrelLayer3Charge->Fill(clustercharge);
                }
            }

            else if(nLayer == 4){
                nClustersBarrelLayer4 += nClustersEv;
                for (SiPixelCluster const& cluster : *it){
                    int clustersize = cluster.size();
                    hClustersSize->Fill(clustersize);
                    hClustersBarrelSize->Fill(clustersize);
                    hClustersBarrelLayer4Size->Fill(clustersize);
                    int clustercharge = cluster.charge();
                    hClustersCharge->Fill(clustercharge);
                    hClustersBarrelCharge->Fill(clustercharge);
                    hClustersBarrelLayer4Charge->Fill(clustercharge);
                }
            }

        }

        else if (subdetid == PixelSubdetector::PixelEndcap) {
            nClustersEndcap += nClustersEv;
            uint32_t ECside = trackerTopology_->pxfSide(id);
            uint32_t nDisk = trackerTopology_->pxfDisk(id);

            if (ECside == 1){
                if(nDisk == 1){
                    nClustersEndcapDiskm1 += nClustersEv;
                    for (SiPixelCluster const& cluster : *it){
                        double clustersize = cluster.size();
                        hClustersSize->Fill(clustersize);
                        hClustersEndcapSize->Fill(clustersize);
                        hClustersEndcapDiskm1Size->Fill(clustersize);
                        double clustercharge = cluster.charge();
                        hClustersCharge->Fill(clustercharge);
                        hClustersEndcapCharge->Fill(clustercharge);
                        hClustersEndcapDiskm1Charge->Fill(clustercharge);
                    }
                }

                else if(nDisk == 2){
                    nClustersEndcapDiskm2 += nClustersEv;
                    for (SiPixelCluster const& cluster : *it){
                        double clustersize = cluster.size();
                        hClustersSize->Fill(clustersize);
                        hClustersEndcapSize->Fill(clustersize);
                        hClustersEndcapDiskm2Size->Fill(clustersize);
                        double clustercharge = cluster.charge();
                        hClustersCharge->Fill(clustercharge);
                        hClustersEndcapCharge->Fill(clustercharge);
                        hClustersEndcapDiskm2Charge->Fill(clustercharge);
                    }
                }

                else if(nDisk == 3){
                    nClustersEndcapDiskm3 += nClustersEv;
                    for (SiPixelCluster const& cluster : *it){
                        double clustersize = cluster.size();
                        hClustersSize->Fill(clustersize);
                        hClustersEndcapSize->Fill(clustersize);
                        hClustersEndcapDiskm3Size->Fill(clustersize);
                        double clustercharge = cluster.charge();
                        hClustersCharge->Fill(clustercharge);
                        hClustersEndcapCharge->Fill(clustercharge);
                        hClustersEndcapDiskm3Charge->Fill(clustercharge);
                    }
                }

            }

            else if (ECside == 2){
                if(nDisk == 1){
                    nClustersEndcapDiskp1 += nClustersEv;
                    for (SiPixelCluster const& cluster : *it){
                        double clustersize = cluster.size();
                        hClustersSize->Fill(clustersize);
                        hClustersEndcapSize->Fill(clustersize);
                        hClustersEndcapDiskp1Size->Fill(clustersize);
                        double clustercharge = cluster.charge();
                        hClustersCharge->Fill(clustercharge);
                        hClustersEndcapCharge->Fill(clustercharge);
                        hClustersEndcapDiskp1Charge->Fill(clustercharge);
                    }
                }
                else if(nDisk == 2){
                    nClustersEndcapDiskp2 += nClustersEv;
                    for (SiPixelCluster const& cluster : *it){
                        double clustersize = cluster.size();
                        hClustersSize->Fill(clustersize);
                        hClustersEndcapSize->Fill(clustersize);
                        hClustersEndcapDiskp2Size->Fill(clustersize);
                        double clustercharge = cluster.charge();
                        hClustersCharge->Fill(clustercharge);
                        hClustersEndcapCharge->Fill(clustercharge);
                        hClustersEndcapDiskp2Charge->Fill(clustercharge);
                    }
                }

                else if(nDisk == 3){
                    nClustersEndcapDiskp3 += nClustersEv;
                    for (SiPixelCluster const& cluster : *it){
                        double clustersize = cluster.size();
                        hClustersSize->Fill(clustersize);
                        hClustersEndcapSize->Fill(clustersize);
                        hClustersEndcapDiskp3Size->Fill(clustersize);
                        double clustercharge = cluster.charge();
                        hClustersCharge->Fill(clustercharge);
                        hClustersEndcapCharge->Fill(clustercharge);
                        hClustersEndcapDiskp3Charge->Fill(clustercharge);
                    }
                }

            }
        }
    }
    hnClusters->Fill(nClusters);
    hnClustersBarrel->Fill(nClustersBarrel);
    hnClustersEndcap->Fill(nClustersEndcap);
    hnClustersBarrelLayer1->Fill(nClustersBarrelLayer1);
    hnClustersBarrelLayer2->Fill(nClustersBarrelLayer2);
    hnClustersBarrelLayer3->Fill(nClustersBarrelLayer3);
    hnClustersBarrelLayer4->Fill(nClustersBarrelLayer4);
    hnClustersEndcapDiskm1->Fill(nClustersEndcapDiskm1);
    hnClustersEndcapDiskm2->Fill(nClustersEndcapDiskm2);
    hnClustersEndcapDiskm3->Fill(nClustersEndcapDiskm3);
    hnClustersEndcapDiskp1->Fill(nClustersEndcapDiskp1);
    hnClustersEndcapDiskp2->Fill(nClustersEndcapDiskp2);
    hnClustersEndcapDiskp3->Fill(nClustersEndcapDiskp3);
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

  hnClusters = ibooker.book1D("nClusters", ";Number of clusters per event;#entries", 50, -0.5, 1000.5);
  hnClustersBarrel = ibooker.book1D("nClustersBarrel", ";Number of clusters per event;#entries", 50, -0.5, 1000.5);
  hnClustersEndcap = ibooker.book1D("nClustersEndcap", ";Number of clusters per event;#entries", 50, -0.5, 1000.5);

  hnClustersBarrelLayer1 = ibooker.book1D("nClustersBarrelLayer1", ";Number of clusters per event;#entries", 50, -0.5, 1000.5);
  hnClustersBarrelLayer2 = ibooker.book1D("nClustersBarrelLayer2", ";Number of clusters per event;#entries", 50, -0.5, 1000.5);
  hnClustersBarrelLayer3 = ibooker.book1D("nClustersBarrelLayer3", ";Number of clusters per event;#entries", 50, -0.5, 1000.5);
  hnClustersBarrelLayer4 = ibooker.book1D("nClustersBarrelLayer4", ";Number of clusters per event;#entries", 50, -0.5, 1000.5);

  hnClustersEndcapDiskm1 = ibooker.book1D("nClustersEndcapDiskm1", ";Number of clusters per event;#entries", 50, -0.5, 1000.5);
  hnClustersEndcapDiskm2 = ibooker.book1D("nClustersEndcapDiskm2", ";Number of clusters per event;#entries", 50, -0.5, 1000.5);
  hnClustersEndcapDiskm3 = ibooker.book1D("nClustersEndcapDiskm3", ";Number of clusters per event;#entries", 50, -0.5, 1000.5);

  hnClustersEndcapDiskp1 = ibooker.book1D("nClustersEndcapDiskp1", ";Number of clusters per event;#entries", 50, -0.5, 1000.5);
  hnClustersEndcapDiskp2 = ibooker.book1D("nClustersEndcapDiskp2", ";Number of clusters per event;#entries", 50, -0.5, 1000.5);
  hnClustersEndcapDiskp3 = ibooker.book1D("nClustersEndcapDiskp3", ";Number of clusters per event;#entries", 50, -0.5, 1000.5);

  hClustersSize = ibooker.book1D("ClustersSize", ";Clusters Size per event;#entries", 50, -0.5, 50.5);
  hClustersBarrelSize = ibooker.book1D("ClustersBarrelSize", ";Clusters Size per event;#entries", 50, -0.5, 50.5);
  hClustersEndcapSize = ibooker.book1D("ClustersEndcapSize", ";Clusters Size per event;#entries", 50, -0.5, 50.5);

  hClustersBarrelLayer1Size = ibooker.book1D("ClustersBarrelLayer1Size", ";Clusters Size per event;#entries", 50, -0.5, 50.5);
  hClustersBarrelLayer2Size = ibooker.book1D("ClustersBarrelLayer2Size", ";Clusters Size per event;#entries", 50, -0.5, 50.5);
  hClustersBarrelLayer3Size = ibooker.book1D("ClustersBarrelLayer3Size", ";Clusters Size per event;#entries", 50, -0.5, 50.5);
  hClustersBarrelLayer4Size = ibooker.book1D("ClustersBarrelLayer4Size", ";Clusters Size per event;#entries", 50, -0.5, 50.5);

  hClustersEndcapDiskm1Size = ibooker.book1D("ClustersEndcapDiskm1Size", ";Clusters Size per event;#entries", 50, -0.5, 50.5);
  hClustersEndcapDiskm2Size = ibooker.book1D("ClustersEndcapDiskm2Size", ";Clusters Size per event;#entries", 50, -0.5, 50.5);
  hClustersEndcapDiskm3Size = ibooker.book1D("ClustersEndcapDiskm3Size", ";Clusters Size per event;#entries", 50, -0.5, 50.5);

  hClustersEndcapDiskp1Size = ibooker.book1D("ClustersEndcapDiskp1Size", ";Clusters Size per event;#entries", 50, -0.5, 50.5);
  hClustersEndcapDiskp2Size = ibooker.book1D("ClustersEndcapDiskp2Size", ";Clusters Size per event;#entries", 50, -0.5, 50.5);
  hClustersEndcapDiskp3Size = ibooker.book1D("ClustersEndcapDiskp3Size", ";Clusters Size per event;#entries", 50, -0.5, 50.5);

  hClustersCharge = ibooker.book1D("ClustersCharge", ";Clusters Charge per event;#entries", 50, -0.5, 300000.5);
  hClustersBarrelCharge = ibooker.book1D("ClustersBarrelCharge", ";Clusters Charge per event;#entries", 50, -0.5, 300000.5);
  hClustersEndcapCharge = ibooker.book1D("ClustersEndcapCharge", ";Clusters Charge per event;#entries", 50, -0.5, 300000.5);

  hClustersBarrelLayer1Charge = ibooker.book1D("ClustersBarrelLayer1Charge", ";Clusters Charge per event;#entries", 50, -0.5, 300000.5);
  hClustersBarrelLayer2Charge = ibooker.book1D("ClustersBarrelLayer2Charge", ";Clusters Charge per event;#entries", 50, -0.5, 300000.5);
  hClustersBarrelLayer3Charge = ibooker.book1D("ClustersBarrelLayer3Charge", ";Clusters Charge per event;#entries", 50, -0.5, 300000.5);
  hClustersBarrelLayer4Charge = ibooker.book1D("ClustersBarrelLayer4Charge", ";Clusters Charge per event;#entries", 50, -0.5, 300000.5);

  hClustersEndcapDiskm1Charge = ibooker.book1D("ClustersEndcapDiskm1Charge", ";Clusters Charge per event;#entries", 50, -0.5, 300000.5);
  hClustersEndcapDiskm2Charge = ibooker.book1D("ClustersEndcapDiskm2Charge", ";Clusters Charge per event;#entries", 50, -0.5, 300000.5);
  hClustersEndcapDiskm3Charge = ibooker.book1D("ClustersEndcapDiskm3Charge", ";Clusters Charge per event;#entries", 50, -0.5, 300000.5);

  hClustersEndcapDiskp1Charge = ibooker.book1D("ClustersEndcapDiskp1Charge", ";Clusters Charge per event;#entries", 50, -0.5, 300000.5);
  hClustersEndcapDiskp2Charge = ibooker.book1D("ClustersEndcapDiskp2Charge", ";Clusters Charge per event;#entries", 50, -0.5, 300000.5);
  hClustersEndcapDiskp3Charge = ibooker.book1D("ClustersEndcapDiskp3Charge", ";Clusters Charge per event;#entries", 50, -0.5, 300000.5);
}

void SiPixelPhase1MonitorClusters::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("pixelClusters", edm::InputTag("siPixelClustersPreSplitting"));
  desc.add<std::string>("TopFolderName", "SiPixelHeterogeneous/PixelClusters");
  descriptions.addWithDefaultLabel(desc);
}
DEFINE_FWK_MODULE(SiPixelPhase1MonitorClusters);
