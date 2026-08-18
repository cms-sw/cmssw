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
// Date: August 2026 (Modified by Lisa Juckett for folder restructure)
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
    unsigned int clusterCounter{0};
  };

  MonitorElement* globalXY_barrel_;
  MonitorElement* globalXY_endcap_;
  MonitorElement* globalRZ_barrel_;
  MonitorElement* globalRZ_endcap_;

  void bookLayerHistos(DQMStore::IBooker& ibooker, uint32_t det_it, std::string& subdir);

  std::map<std::string, ClusterMEs> layerMEs_;
  enum Level { IT = 1, SUBSTRUCTURE, SHELL, ENDCAP_RING, ENDCAP_WHEEL, LAYER };
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
    for (const auto& clusterItr : DSVItr) {
      MeasurementPoint mpCluster(clusterItr.x(), clusterItr.y());
      Local3DPoint localPosCluster = geomDetUnit->topology().localPosition(mpCluster);
      Global3DPoint globalPosCluster = geomDetUnit->surface().toGlobal(localPosCluster);
      double gx = globalPosCluster.x() * 10.;
      double gy = globalPosCluster.y() * 10.;
      double gz = globalPosCluster.z() * 10.;
      double gr = globalPosCluster.perp() * 10.;

      // Fill non-layer histos
      if (geomDetUnit->subDetector() == GeomDetEnumerators::SubDetector::P2PXB) {
        globalXY_barrel_->Fill(gx, gy);
        globalRZ_barrel_->Fill(gz, gr);
      } else if (geomDetUnit->subDetector() == GeomDetEnumerators::SubDetector::P2PXEC) {
        globalXY_endcap_->Fill(gx, gy);
        globalRZ_endcap_->Fill(gz, gr);
      }
      for (enum Level fillingDepth = IT; fillingDepth <= LAYER; fillingDepth = Level(fillingDepth + 1)) {
        // Skip filling for barrel detIds on endcap-only depths
        if ((fillingDepth == ENDCAP_RING || fillingDepth == ENDCAP_WHEEL) &&
            DetId(detId).subdetId() == PixelSubdetector::PixelBarrel)
          continue;
        std::string folderkey = phase2tkutil::getHistoId(detId, tTopo_, detPos.phi(), fillingDepth, false);
        auto local_mesIT = layerMEs_.find(folderkey);
        if (local_mesIT == layerMEs_.end())
          continue;
        ClusterMEs& local_mes = local_mesIT->second;

        local_mes.ClusterSize->Fill(clusterItr.size());
        local_mes.ClusterSizeX->Fill(clusterItr.sizeX());
        local_mes.ClusterSizeY->Fill(clusterItr.sizeY());
        local_mes.ClusterCharge->Fill(clusterItr.charge());
        local_mes.clusterCounter++;
      }
    }
  }
  for (auto& it : layerMEs_) {
    ClusterMEs& local_mes = it.second;
    if (local_mes.nClusters)
      local_mes.nClusters->Fill(local_mes.clusterCounter);
    local_mes.clusterCounter = 0;
  }
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
  ibooker.setCurrentFolder(folder_name.str() + "/Positions");
  edm::LogInfo("Phase2ITMonitorCluster") << " Booking Histograms in: " << folder_name.str();

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
      edm::LogInfo("Phase2ITMonitorCluster")
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
  for (enum Level bookingDepth = IT; bookingDepth <= LAYER; bookingDepth = Level(bookingDepth + 1)) {
    // Skip booking for barrel det_ids in endcap-only depths
    if ((bookingDepth == ENDCAP_RING || bookingDepth == ENDCAP_WHEEL) &&
        DetId(det_id).subdetId() == PixelSubdetector::PixelBarrel)
      continue;

    std::string folderName = phase2tkutil::getHistoId(det_id, tTopo_, detPos.phi(), bookingDepth, false);
    std::string prettyName = phase2tkutil::getHistoId(det_id, tTopo_, detPos.phi(), bookingDepth, true);

    std::map<std::string, ClusterMEs>::iterator pos = layerMEs_.find(folderName);

    if (pos == layerMEs_.end()) {
      ibooker.cd();
      ibooker.setCurrentFolder(subdir + "/" + folderName);

      edm::LogInfo("Phase2ITMonitorCluster") << " Booking Histograms in: " << subdir + "/" + folderName;
      ClusterMEs local_mes;

      local_mes.nClusters = phase2tkutil::book1DFromPSet(
          config_.getParameter<edm::ParameterSet>("NClustersLayer"), ibooker, prettyName, bookingDepth);

      local_mes.ClusterSize =
          phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("ClusterSize"), ibooker, prettyName);

      local_mes.ClusterSizeX =
          phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("ClusterSizeX"), ibooker, prettyName);

      local_mes.ClusterSizeY =
          phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("ClusterSizeY"), ibooker, prettyName);

      local_mes.ClusterCharge =
          phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("ClusterCharge"), ibooker, prettyName);

      layerMEs_.emplace(folderName, local_mes);
    }
  }
}

void Phase2ITMonitorCluster::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // clusterITMonitor
  edm::ParameterSetDescription desc;
  //Global Histos
  phase2tkutil::add2DDesc(desc,
                          "GlobalPositionRZ_PXB",
                          "Clusters_Global_Position_RZ_IT_barrel",
                          "Clusters_Global_Position_RZ_IT_barrel",
                          "z [mm]",
                          "r [mm]",
                          1500,
                          -3000.0,
                          3000.0,
                          300,
                          0.0,
                          300.0);
  phase2tkutil::add2DDesc(desc,
                          "GlobalPositionXY_PXB",
                          "Clusters_Global_Position_XY_IT_barrel",
                          "Clusters_Global_Position_XY_IT_barrel",
                          "x [mm]",
                          "y [mm]",
                          600,
                          -300.0,
                          300.0,
                          600,
                          -300.0,
                          300.0);
  phase2tkutil::add2DDesc(desc,
                          "GlobalPositionRZ_PXEC",
                          "Clusters_Global_Position_RZ_IT_endcap",
                          "Clusters_Global_Position_RZ_IT_endcap",
                          "z [mm]",
                          "r [mm]",
                          1500,
                          -3000.0,
                          3000.0,
                          300,
                          0.0,
                          300.0);
  phase2tkutil::add2DDesc(desc,
                          "GlobalPositionXY_PXEC",
                          "Clusters_Global_Position_XY_IT_endcap",
                          "Clusters_Global_Position_XY_IT_endcap",
                          "x [mm]",
                          "y [mm]",
                          600,
                          -300.0,
                          300.0,
                          600,
                          -300.0,
                          300.0);

  //Per layer/ring histos
  phase2tkutil::add1DDesc(desc,
                          "NClustersLayer",
                          "Num_Clusters_Per_Event",
                          "Number Of Clusters per event in {}",
                          "Number of Clusters per event",
                          "Number of events",
                          150,
                          0.0,
                          250000.0);
  phase2tkutil::add1DDesc(desc,
                          "ClusterCharge",
                          "Cluster_Charge",
                          "Cluster charge in {}",
                          "Cluster charge",
                          "Number of clusters",
                          100,
                          0.0,
                          100000.0);
  phase2tkutil::add1DDesc(
      desc, "ClusterSize", "Cluster_Size", "Cluster size in {}", "Cluster size", "Number of clusters", 31, -0.5, 30.5);
  phase2tkutil::add1DDesc(desc,
                          "ClusterSizeY",
                          "Cluster_Size_Y",
                          "Cluster size Y in {}",
                          "Cluster size Y",
                          "Number of clusters",
                          31,
                          -0.5,
                          30.5);
  phase2tkutil::add1DDesc(desc,
                          "ClusterSizeX",
                          "Cluster_Size_X",
                          "Cluster size X in {}",
                          "Cluster size X",
                          "Number of clusters",
                          31,
                          -0.5,
                          30.5);

  desc.add<std::string>("TopFolderName", "InnerTracker");
  desc.add<edm::InputTag>("InnerPixelClusterSource", edm::InputTag("siPixelClusters"));
  descriptions.add("Phase2ITMonitorCluster", desc);
}

DEFINE_FWK_MODULE(Phase2ITMonitorCluster);
