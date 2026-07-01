// -*- C++ -*-
//bookLayer
// Package:    Phase2ITMonitorCluster
// Class:      Phase2ITMonitorCluster
//
/**\class Phase2ITMonitorCluster Phase2ITMonitorCluster.cc 

 Description: DQM plots tracker clusters. 

*/
//
// Author: Gabriel Ramirez
// Date: May 23, 2020
#include <memory>
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
// DQM Histograming
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

class Phase2ITMonitorCluster : public DQMEDAnalyzer {
public:
  explicit Phase2ITMonitorCluster(const edm::ParameterSet&);
  ~Phase2ITMonitorCluster() override;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  void dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  struct ClusterMEs {
    MonitorElement* nClusters = nullptr;
    MonitorElement* ClusterSize = nullptr;
    MonitorElement* ClusterSizeX = nullptr;
    MonitorElement* ClusterSizeY = nullptr;
    MonitorElement* ClusterCharge = nullptr;
    MonitorElement* XYGlobalPositionMapPixel = nullptr;
    MonitorElement* XYLocalPositionMapPixel = nullptr;
  };

  MonitorElement* numberClusters_;
  MonitorElement* globalXY_barrel_;
  MonitorElement* globalXY_endcap_;
  MonitorElement* globalRZ_barrel_;
  MonitorElement* globalRZ_endcap_;

  void bookLayerHistos(DQMStore::IBooker& ibooker, uint32_t det_it, std::string& subdir);

  std::map<std::string, ClusterMEs> layerMEs_;
  edm::ParameterSet config_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster>> itPixelClusterToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  const TrackerGeometry* tkGeom_ = nullptr;
  const TrackerTopology* tTopo_ = nullptr;
};
#include "DQM/SiTrackerPhase2/interface/TrackerPhase2DQMUtil.h"
//
// constructors
//
Phase2ITMonitorCluster::Phase2ITMonitorCluster(const edm::ParameterSet& iConfig)
    : config_(iConfig),
      itPixelClusterToken_(consumes<edmNew::DetSetVector<SiPixelCluster>>(
          config_.getParameter<edm::InputTag>("InnerPixelClusterSource"))),
      geomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>()),
      topoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd, edm::Transition::BeginRun>()) {
  edm::LogInfo("Phase2ITMonitorCluster") << ">>> Construct Phase2ITMonitorCluster ";
}

Phase2ITMonitorCluster::~Phase2ITMonitorCluster() {
  edm::LogInfo("Phase2ITMonitorCluster") << ">>> Destroy Phase2ITMonitorCluster ";
}
//
// -- DQM Begin Run
void Phase2ITMonitorCluster::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  tkGeom_ = &iSetup.getData(geomToken_);
  tTopo_ = &iSetup.getData(topoToken_);
}

//
// -- Analyze
//
void Phase2ITMonitorCluster::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Getting the clusters
  const auto& itPixelClusterHandle = iEvent.getHandle(itPixelClusterToken_);

  if (!itPixelClusterHandle.isValid()) {
    edm::LogWarning("Phase2ITMonitorCluster") << "No SiPixelCluster Collection found in the event. Skipping!";
    return;
  }

  // Number of clusters
  std::map<std::string, unsigned int> nClsmap;
  unsigned int nclusGlobal = 0;
  for (const auto& DSVItr : *itPixelClusterHandle) {
    uint32_t rawid(DSVItr.detId());
    DetId detId(rawid);
    const GeomDet* geomDet = tkGeom_->idToDet(detId);
    if (!geomDet)
      continue;
    const GeomDetUnit* geomDetUnit(tkGeom_->idToDetUnit(detId));
    if (!geomDetUnit)
      continue;
    GlobalPoint detPos = geomDet->surface().toGlobal(Local2DPoint(0, 0));
    nclusGlobal += DSVItr.size();
    int nClus = 0;
    for (const auto& clusterItr : DSVItr) {
      MeasurementPoint mpCluster(clusterItr.x(), clusterItr.y());
      Local3DPoint localPosCluster = geomDetUnit->topology().localPosition(mpCluster);
      Global3DPoint globalPosCluster = geomDetUnit->surface().toGlobal(localPosCluster);
      double gx = globalPosCluster.x() * 10.;
      double gy = globalPosCluster.y() * 10.;
      double gz = globalPosCluster.z() * 10.;
      double gr = globalPosCluster.perp() * 10.;
      nClus++;

      // Fill non-layer histos
      if (geomDetUnit->subDetector() == GeomDetEnumerators::SubDetector::P2PXB) {
        globalXY_barrel_->Fill(gx, gy);
        globalRZ_barrel_->Fill(gz, gr);
      } else if (geomDetUnit->subDetector() == GeomDetEnumerators::SubDetector::P2PXEC) {
        globalXY_endcap_->Fill(gx, gy);
        globalRZ_endcap_->Fill(gz, gr);
      }
      // Workaround for filling histograms in both Ring<> and Wheel<>
      bool isEndcap = (detId.subdetId() != PixelSubdetector::PixelBarrel);
      for (int booking = 1; booking < 2 + isEndcap; booking++) {
        // Will loop twice if the module is an EndCap module
        // the default key divides endcaps into F/EPixs and Rings
        // in second loop endcaps will be divided into F/EPix and Wheels
        std::string folderkey = (booking == 2 ? phase2tkutil::getITHistoWheelId(detId, tTopo_, detPos.phi())
                                              : phase2tkutil::getITHistoId(detId, tTopo_, detPos.phi()));

        auto local_mesIT = layerMEs_.find(folderkey);
        if (local_mesIT == layerMEs_.end())
          continue;
        ClusterMEs& local_mes = local_mesIT->second;
        if (local_mes.XYGlobalPositionMapPixel)  //make this optional
          local_mes.XYGlobalPositionMapPixel->Fill(globalPosCluster.z(), globalPosCluster.perp());
        local_mes.XYLocalPositionMapPixel->Fill(localPosCluster.x(), localPosCluster.y());
        local_mes.ClusterSize->Fill(clusterItr.size());
        local_mes.ClusterSizeX->Fill(clusterItr.sizeX());
        local_mes.ClusterSizeY->Fill(clusterItr.sizeY());
        local_mes.ClusterCharge->Fill(clusterItr.charge());

        if (nClus == int(DSVItr.size())) {
          // Reached the end of clusters in this Det
          // Fill any histos that should only be filled once per det

          // initialize the nhit counters if they don't exist for this layer
          auto counterDet = nClsmap.find(folderkey);
          if (counterDet == nClsmap.end()) {
            nClsmap.emplace(folderkey, DSVItr.size());
          } else
            counterDet->second += DSVItr.size();
        }
      }
    }
  }

  for (const auto& it : nClsmap) {
    if (layerMEs_.find(it.first) == layerMEs_.end())
      continue;
    layerMEs_[it.first].nClusters->Fill(it.second);
  }
  numberClusters_->Fill(nclusGlobal);  //global histo of #clusters
}

//
// -- Book Histograms
//
void Phase2ITMonitorCluster::bookHistograms(DQMStore::IBooker& ibooker,
                                            edm::Run const& iRun,
                                            edm::EventSetup const& iSetup) {
  std::string top_folder = config_.getParameter<std::string>("TopFolderName");
  std::stringstream folder_name;

  ibooker.cd();
  folder_name << top_folder << "/";
  ibooker.setCurrentFolder(folder_name.str());

  edm::LogInfo("Phase2ITMonitorCluster") << " Booking Histograms in: " << folder_name.str();

  numberClusters_ = phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("GlobalNClusters"), ibooker);

  ibooker.setCurrentFolder(folder_name.str() + "/Positions");
  globalXY_barrel_ =
      phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("GlobalPositionXY_PXB"), ibooker);

  globalRZ_barrel_ =
      phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("GlobalPositionRZ_PXB"), ibooker);

  globalXY_endcap_ =
      phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("GlobalPositionXY_PXEC"), ibooker);

  globalRZ_endcap_ =
      phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("GlobalPositionRZ_PXEC"), ibooker);

  //Now book layer wise histos
  edm::ESWatcher<TrackerDigiGeometryRecord> theTkDigiGeomWatcher;
  if (theTkDigiGeomWatcher.check(iSetup)) {
    for (auto const& det_u : tkGeom_->detUnits()) {
      //Always check TrackerNumberingBuilder before changing this part
      if (!(det_u->subDetector() == GeomDetEnumerators::SubDetector::P2PXB ||
            det_u->subDetector() == GeomDetEnumerators::SubDetector::P2PXEC))
        continue;
      unsigned int detId_raw = det_u->geographicalId().rawId();
      GlobalPoint detPos = det_u->surface().toGlobal(Local2DPoint(0, 0));
      edm::LogInfo("Phase2ITMonitorRecHit")
          << "Detid:" << detId_raw << "\tsubdet=" << det_u->subDetector()
          << "\t key=" << phase2tkutil::getITHistoId(detId_raw, tTopo_, detPos.phi()) << std::endl;
      bookLayerHistos(ibooker, detId_raw, top_folder);
    }
  }
}

//////////////////Layer Histo/////////////////////////////////
void Phase2ITMonitorCluster::bookLayerHistos(DQMStore::IBooker& ibooker, uint32_t det_id, std::string& subdir) {
  const GeomDet* geomDet = tkGeom_->idToDet(det_id);
  GlobalPoint detPos = geomDet->surface().toGlobal(Local2DPoint(0, 0));
  // Workaround for booking same histogram for Ring<> and Wheel<>
  bool isEndcap = (DetId(det_id).subdetId() != PixelSubdetector::PixelBarrel);
  for (int booking = 1; booking < 2 + isEndcap; booking++) {
    // Will loop twice if the module is an EndCap module
    // By default, the "key" divides endcaps into F/Epix and Rings
    // During first loop, the default key is used
    // In the second loop, the Wheel key is used
    // all layer-wise histograms will be booked in Wheels as well as Rings
    std::string folderName = (booking == 2 ? phase2tkutil::getITHistoWheelId(det_id, tTopo_, detPos.phi())
                                           : phase2tkutil::getITHistoId(det_id, tTopo_, detPos.phi()));

    if (folderName.empty())
      return;

    std::map<std::string, ClusterMEs>::iterator pos = layerMEs_.find(folderName);

    if (pos == layerMEs_.end()) {
      ibooker.cd();
      ibooker.setCurrentFolder(subdir + "/" + folderName);

      edm::LogInfo("Phase2ITMonitorCluster") << " Booking Histograms in: " << subdir + "/" + folderName;

      ClusterMEs local_mes;

      local_mes.nClusters =
          phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("NClustersLayer"), ibooker);

      local_mes.ClusterSize =
          phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("ClusterSize"), ibooker);

      local_mes.ClusterSizeX =
          phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("ClusterSizeX"), ibooker);

      local_mes.ClusterSizeY =
          phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("ClusterSizeY"), ibooker);

      local_mes.ClusterCharge =
          phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("ClusterCharge"), ibooker);

      local_mes.XYLocalPositionMapPixel =
          phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("LocalPositionXY"), ibooker);

      local_mes.XYGlobalPositionMapPixel =
          phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("GlobalPositionXY_perlayer"), ibooker);

      layerMEs_.emplace(folderName, local_mes);
    }
  }
}

void Phase2ITMonitorCluster::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // clusterITMonitor
  edm::ParameterSetDescription desc;
  //Global Histos
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Num_Clusters");
    psd0.add<std::string>("title", "NumberClusters;Number of Clusters;");
    psd0.add<double>("xmin", 0.0);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 300000.0);
    psd0.add<int>("NxBins", 150);
    desc.add<edm::ParameterSetDescription>("GlobalNClusters", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Clusters_Global_Position_RZ_IT_barrel");
    psd0.add<std::string>("title", "Clusters_Global_Position_RZ_IT_barrel;z [mm];r [mm]");
    psd0.add<double>("ymax", 300.0);
    psd0.add<int>("NxBins", 1500);
    psd0.add<int>("NyBins", 300);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 3000.0);
    psd0.add<double>("xmin", -3000.0);
    psd0.add<double>("ymin", 0.0);
    desc.add<edm::ParameterSetDescription>("GlobalPositionRZ_PXB", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Clusters_Global_Position_XY_IT_barrel");
    psd0.add<std::string>("title", "Clusters_Global_Position_XY_IT_barrel;x [mm];y [mm];");
    psd0.add<double>("ymax", 300.0);
    psd0.add<int>("NxBins", 600);
    psd0.add<int>("NyBins", 600);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 300.0);
    psd0.add<double>("xmin", -300.0);
    psd0.add<double>("ymin", -300.0);
    desc.add<edm::ParameterSetDescription>("GlobalPositionXY_PXB", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Clusters_Global_Position_RZ_IT_endcap");
    psd0.add<std::string>("title", "Clusters_Global_Position_RZ_IT_endcap;z [mm];r [mm]");
    psd0.add<double>("ymax", 300.0);
    psd0.add<int>("NxBins", 1500);
    psd0.add<int>("NyBins", 300);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 3000.0);
    psd0.add<double>("xmin", -3000.0);
    psd0.add<double>("ymin", 0.0);
    desc.add<edm::ParameterSetDescription>("GlobalPositionRZ_PXEC", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Clusters_Global_Position_XY_IT_endcap");
    psd0.add<std::string>("title", "Clusters_Global_Position_XY_IT_endcap; x [mm]; y [mm]");
    psd0.add<double>("ymax", 300.0);
    psd0.add<int>("NxBins", 600);
    psd0.add<int>("NyBins", 600);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 300.0);
    psd0.add<double>("xmin", -300.0);
    psd0.add<double>("ymin", -300.0);
    desc.add<edm::ParameterSetDescription>("GlobalPositionXY_PXEC", psd0);
  }
  //Local histos
  //Per layer/ring histos
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Num_Clusters_Layer");
    psd0.add<std::string>("title", "NumberOfClusters;Number of Clusters;");
    psd0.add<double>("xmin", 0.0);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 20000.0);
    psd0.add<int>("NxBins", 150);
    desc.add<edm::ParameterSetDescription>("NClustersLayer", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Cluster_Charge");
    psd0.add<std::string>("title", ";Cluster charge;");
    psd0.add<double>("xmin", 0.);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 100000.);
    psd0.add<int>("NxBins", 100);
    desc.add<edm::ParameterSetDescription>("ClusterCharge", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Cluster_Size");
    psd0.add<std::string>("title", ";Cluster size;");
    psd0.add<double>("xmin", -0.5);
    psd0.add<double>("xmax", 30.5);
    psd0.add<int>("NxBins", 31);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("ClusterSize", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Cluster_Size_Y");
    psd0.add<std::string>("title", ";Cluster sizeY;");
    psd0.add<double>("xmin", -0.5);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 30.5);
    psd0.add<int>("NxBins", 31);
    desc.add<edm::ParameterSetDescription>("ClusterSizeY", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Cluster_Size_X");
    psd0.add<std::string>("title", ";Cluster sizeX;");
    psd0.add<double>("xmin", -0.5);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 30.5);
    psd0.add<int>("NxBins", 31);
    desc.add<edm::ParameterSetDescription>("ClusterSizeX", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Local_Cluster_Position_XY");
    psd0.add<std::string>("title", "Local_ClusterPosition_XY; x; y");
    psd0.add<double>("ymax", 0.0);
    psd0.add<int>("NxBins", 500);
    psd0.add<int>("NyBins", 500);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 0.0);
    psd0.add<double>("xmin", 0.0);
    psd0.add<double>("ymin", 0.0);
    desc.add<edm::ParameterSetDescription>("LocalPositionXY", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "GlobalPositionXY_perlayer");
    psd0.add<std::string>("title", "GlobalClusterPositionXY_perlayer;x [mm];y [mm];");
    psd0.add<double>("ymax", 300.0);
    psd0.add<int>("NxBins", 600);
    psd0.add<int>("NyBins", 600);
    psd0.add<bool>("switch", false);
    psd0.add<double>("xmax", 300.0);
    psd0.add<double>("xmin", -300.0);
    psd0.add<double>("ymin", -300.0);
    desc.add<edm::ParameterSetDescription>("GlobalPositionXY_perlayer", psd0);
  }

  desc.add<std::string>("TopFolderName", "InnerTracker");
  desc.add<edm::InputTag>("InnerPixelClusterSource", edm::InputTag("siPixelClusters"));
  descriptions.add("Phase2ITMonitorCluster", desc);
}

DEFINE_FWK_MODULE(Phase2ITMonitorCluster);
