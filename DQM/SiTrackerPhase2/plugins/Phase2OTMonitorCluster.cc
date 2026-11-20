// -*- C++ -*-
//bookLayer
// Package:    Phase2OTMonitorCluster
// Class:      Phase2OTMonitorCluster
//
/**\class Phase2OTMonitorCluster Phase2OTMonitorCluster.cc 

 Description: Validation plots tracker clusters. 

*/
//
// Author: Gabriel Ramirez
// Date: May 23, 2020
// Date: August 2026 (modified by Lisa Juckett for folder restructure)
#include <memory>
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
// DQM Histograming
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

class Phase2OTMonitorCluster : public DQMEDAnalyzer {
public:
  explicit Phase2OTMonitorCluster(const edm::ParameterSet&);
  ~Phase2OTMonitorCluster() override;
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup) override;
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  void dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  struct ClusterMEs {
    MonitorElement* nClusters_P = nullptr;
    MonitorElement* ClusterSize_P = nullptr;

    MonitorElement* nClusters_S = nullptr;
    MonitorElement* ClusterSize_S = nullptr;

    std::vector<MonitorElement*> PositionOfClusters_2S;  // CRACK ONLY
    std::vector<MonitorElement*> ModuleXCorrelation_2S;  // CRACK ONLY
    std::vector<std::vector<std::pair<int, float>>> ClusterX_vs_sensor_Module;
    std::vector<MonitorElement*> PositionOfClusters_2SLadder;  // Barrel and LAYER only
    unsigned int clusterCounterP = 0;
    unsigned int clusterCounterS = 0;
  };
  MonitorElement* globalXY_P_;
  MonitorElement* globalRZ_P_;
  MonitorElement* globalXY_S_;
  MonitorElement* globalRZ_S_;
  MonitorElement* numberClusters_Barrel_;
  MonitorElement* crackOverview_;
  MonitorElement* clusterSize_map_CRACK;
  MonitorElement* numClusters_map_CRACK;

  void fillOTHistos(const edm::Event& iEvent);

  void bookLayerHistos(DQMStore::IBooker& ibooker, uint32_t det_it, std::string& subdir);

  std::map<int, ClusterMEs> layerMEs_;

  edm::ParameterSet config_;
  edm::EDGetTokenT<Phase2TrackerCluster1DCollectionNew> clustersToken_;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  const TrackerGeometry* tkGeom_ = nullptr;
  const TrackerTopology* tTopo_ = nullptr;
};
#include "DQM/SiTrackerPhase2/interface/TrackerPhase2DQMUtil.h"
//
// constructors
//
Phase2OTMonitorCluster::Phase2OTMonitorCluster(const edm::ParameterSet& iConfig)
    : config_(iConfig),
      clustersToken_(consumes<Phase2TrackerCluster1DCollectionNew>(config_.getParameter<edm::InputTag>("clusterSrc"))),
      geomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>()),
      topoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd, edm::Transition::BeginRun>()) {
  edm::LogInfo("Phase2OTMonitorCluster") << ">>> Construct Phase2OTMonitorCluster ";
}

Phase2OTMonitorCluster::~Phase2OTMonitorCluster() {
  edm::LogInfo("Phase2OTMonitorCluster") << ">>> Destroy Phase2OTMonitorCluster ";
}
//
// -- DQM Begin Run
void Phase2OTMonitorCluster::dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) {
  tkGeom_ = &iSetup.getData(geomToken_);
  tTopo_ = &iSetup.getData(topoToken_);
}
//
// -- Analyze
//
void Phase2OTMonitorCluster::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Getting the clusters
  const auto& clusterHandle = iEvent.getHandle(clustersToken_);

  if (!clusterHandle.isValid()) {
    edm::LogWarning("Phase2OTMonitorCluster") << "No Phase2TrackerCluster1D Collection found in the event. Skipping!";
    return;
  }

  for (const auto& DSVItr : *clusterHandle) {
    // Getting the id of detector unit
    uint32_t rawid(DSVItr.detId());
    DetId detId(rawid);
    const GeomDetUnit* geomDetUnit(tkGeom_->idToDetUnit(detId));
    if (!geomDetUnit)
      continue;

    TrackerGeometry::ModuleType mType = tkGeom_->getDetectorType(detId);
    unsigned int module = tTopo_->module(rawid);
    unsigned int ladder = tTopo_->tobRod(rawid);
    int layer = tTopo_->layer(rawid);
    int isLowerSensor = tTopo_->isLower(rawid);

    // Get the histograms for each depth of this detId
    std::vector<ClusterMEs*> thisDetDepthMEs;
    for (int fillingDepth = phase2tkutil::WHOLE_TK; fillingDepth <= phase2tkutil::LAYER; fillingDepth++) {
      // Skip filling for barrel detIds on endcap-only depths
      if ((fillingDepth >= phase2tkutil::SHELL && fillingDepth < phase2tkutil::LAYER) &&
          DetId(detId).subdetId() == SiStripSubdetector::TOB)
        continue;
      int key = phase2tkutil::getNumericHistoId(detId.rawId(), tTopo_, 0, fillingDepth);

      if (layerMEs_.find(key) == layerMEs_.end())
        continue;
      ClusterMEs* depthLayerMEs = &layerMEs_.find(key)->second;

      thisDetDepthMEs.push_back(depthLayerMEs);
    }

    for (const auto& clusterItr : DSVItr) {
      MeasurementPoint mpCluster(clusterItr.center(), clusterItr.column() + 0.5);
      Local3DPoint localPosCluster = geomDetUnit->topology().localPosition(mpCluster);
      Global3DPoint globalPosCluster = geomDetUnit->surface().toGlobal(localPosCluster);
      double gx = globalPosCluster.x();
      double gy = globalPosCluster.y();
      double gz = globalPosCluster.z();
      double gr = globalPosCluster.perp();
      int topOrBottomColumn = 0;

      // CRACK is viewed from behind, so to align plots with what is seen in real life, modules are flipped
      if (crackOverview_)
        module = std::abs(int(module - 13));
      if (mType == TrackerGeometry::ModuleType::Ph2PSP) {
        globalXY_P_->Fill(gx, gy);
        globalRZ_P_->Fill(gz, gr);
      } else if (mType == TrackerGeometry::ModuleType::Ph2PSS || mType == TrackerGeometry::ModuleType::Ph2SS) {
        globalXY_S_->Fill(gx, gy);
        globalRZ_S_->Fill(gz, gr);
      }
      if (detId.subdetId() == SiStripSubdetector::TOB) {
        if (clusterSize_map_CRACK)
          clusterSize_map_CRACK->Fill(module, layer, clusterItr.size());
        if (numberClusters_Barrel_)
          numberClusters_Barrel_->Fill(layer);
        if (mType == TrackerGeometry::ModuleType::Ph2SS) {
          //If column is on the bottom of the sensor, *-1 to distinguish it from top
          topOrBottomColumn = (isLowerSensor ? (clusterItr.column() + 1) * -1 : (clusterItr.column() + 1));
          if (crackOverview_)
            crackOverview_->Fill(module, layer + 0.05 - (module % 2 * 0.1));
          if (numClusters_map_CRACK)
            numClusters_map_CRACK->Fill(module, layer);
        }
      }

      // Iterate through found depths for this det
      for (auto depthMEs : thisDetDepthMEs) {
        if (mType == TrackerGeometry::ModuleType::Ph2PSP) {
          // Pixels
          if (depthMEs->ClusterSize_P)
            depthMEs->ClusterSize_P->Fill(clusterItr.size());
          depthMEs->clusterCounterP++;
        } else if (mType == TrackerGeometry::ModuleType::Ph2PSS || mType == TrackerGeometry::ModuleType::Ph2SS) {
          // Strips
          if (depthMEs->ClusterSize_S)
            depthMEs->ClusterSize_S->Fill(clusterItr.size());
          depthMEs->clusterCounterS++;
          if (mType == TrackerGeometry::ModuleType::Ph2SS) {
            if (!depthMEs->PositionOfClusters_2S.empty())
              if (depthMEs->PositionOfClusters_2S[module - 1])
                depthMEs->PositionOfClusters_2S[module - 1]->Fill(clusterItr.center(), topOrBottomColumn);

            if (!depthMEs->ClusterX_vs_sensor_Module.empty())
              depthMEs->ClusterX_vs_sensor_Module[module - 1].push_back(std::pair(isLowerSensor, localPosCluster.x()));

            if (!depthMEs->PositionOfClusters_2SLadder.empty()) {
              if (depthMEs->PositionOfClusters_2SLadder[ladder - 1]) {
                int signedModule = module;
                // CRACK has numbers 1 to 12 while Tracker has 1 to 24
                // Adapt module numbers from 1 to 24 into -12 to +12
                if (!crackOverview_)
                  signedModule = module <= 12 ? module - 13 : module - 12;
                depthMEs->PositionOfClusters_2SLadder[ladder - 1]->Fill(signedModule, topOrBottomColumn);
              }
            }
          }
        }
      }
    }
  }
  // After all clusters in event are processed
  for (auto& it : layerMEs_) {
    ClusterMEs& local_mes = it.second;
    if (local_mes.nClusters_P)
      local_mes.nClusters_P->Fill(local_mes.clusterCounterP);
    local_mes.clusterCounterP = 0;
    if (local_mes.nClusters_S)
      local_mes.nClusters_S->Fill(local_mes.clusterCounterS);
    local_mes.clusterCounterS = 0;

    if (!local_mes.ClusterX_vs_sensor_Module.empty() && !local_mes.ModuleXCorrelation_2S.empty()) {
      for (int module = 0; module < int(local_mes.ClusterX_vs_sensor_Module.size()); module++) {
        if (!local_mes.ClusterX_vs_sensor_Module[module].empty()) {
          std::vector<float> clustersUpper, clustersLower;
          for (auto cluster : local_mes.ClusterX_vs_sensor_Module[module]) {
            if (cluster.first == 1)
              clustersLower.push_back(cluster.second);
            else
              clustersUpper.push_back(cluster.second);
          }
          if (!clustersUpper.empty() && !clustersLower.empty()) {
            for (int index = 0; index < int(std::min(clustersLower.size(), clustersUpper.size())); index++) {
              if (local_mes.ModuleXCorrelation_2S[module]) {
                local_mes.ModuleXCorrelation_2S[module]->Fill(clustersUpper[index], clustersLower[index]);
              }
            }
          }
          local_mes.ClusterX_vs_sensor_Module[module] = {};
        }
      }
    }
  }
}

//
// -- Book Histograms
//
void Phase2OTMonitorCluster::bookHistograms(DQMStore::IBooker& ibooker,
                                            edm::Run const& iRun,
                                            edm::EventSetup const& iSetup) {
  std::string top_folder = config_.getParameter<std::string>("TopFolderName");
  ibooker.cd();
  ibooker.setCurrentFolder(top_folder);
  edm::LogInfo("Phase2OTMonitorCluster") << " Booking Histograms in: " << top_folder;

  edm::ParameterSet crack_overview_par = config_.getParameter<edm::ParameterSet>("CrackOverview");
  if (crack_overview_par.getParameter<bool>("switch")) {
    crackOverview_ = ibooker.book2DPoly(crack_overview_par.getParameter<std::string>("name"),
                                        crack_overview_par.getParameter<std::string>("title"),
                                        crack_overview_par.getParameter<double>("xmin"),
                                        crack_overview_par.getParameter<double>("xmax"),
                                        crack_overview_par.getParameter<double>("ymin"),
                                        crack_overview_par.getParameter<double>("ymax"));
    if (crackOverview_->getTH2Poly()->GetNumberOfBins() == 0) {
      std::vector<double> crack_overview_bins = config_.getParameter<std::vector<double>>("CrackOverview_Bins");
      for (int i = 0; i < int(crack_overview_bins.size()); i = i + 4) {
        crackOverview_->addBin(
            crack_overview_bins[i], crack_overview_bins[i + 1], crack_overview_bins[i + 2], crack_overview_bins[i + 3]);
      }
    }
    crackOverview_->getTH2Poly()->SetStats(false);
    crackOverview_->setOption("z0");

  } else
    crackOverview_ = nullptr;

  edm::ParameterSet crack_cluster_size_map = config_.getParameter<edm::ParameterSet>("CrackClusterSizeMap");
  if (crack_cluster_size_map.getParameter<bool>("switch")) {
    clusterSize_map_CRACK = ibooker.bookProfile2D(crack_cluster_size_map.getParameter<std::string>("name"),
                                                  crack_cluster_size_map.getParameter<std::string>("title"),
                                                  crack_cluster_size_map.getParameter<int32_t>("NxBins"),
                                                  crack_cluster_size_map.getParameter<double>("xmin"),
                                                  crack_cluster_size_map.getParameter<double>("xmax"),
                                                  crack_cluster_size_map.getParameter<int32_t>("NyBins"),
                                                  crack_cluster_size_map.getParameter<double>("ymin"),
                                                  crack_cluster_size_map.getParameter<double>("ymax"),
                                                  0,  // zmin (ignored)
                                                  0   // zmax (ignored)
    );
    clusterSize_map_CRACK->getTProfile2D()->SetStats(false);
    clusterSize_map_CRACK->setOption("colz");

    crack_cluster_size_map.addParameter<std::string>("name", "Crack_NumClusters");
    crack_cluster_size_map.addParameter<std::string>("title", "Number of clusters by module and layer;Module;Layer");
    numClusters_map_CRACK = phase2tkutil::book2DFromPSet(crack_cluster_size_map, ibooker);
  } else {
    clusterSize_map_CRACK = nullptr;
    numClusters_map_CRACK = nullptr;
  }

  ibooker.setCurrentFolder(top_folder + "/Positions/");
  globalXY_P_ = phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("GlobalPositionXY_P"), ibooker);

  globalRZ_P_ = phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("GlobalPositionRZ_P"), ibooker);

  globalXY_S_ = phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("GlobalPositionXY_S"), ibooker);

  globalRZ_S_ = phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("GlobalPositionRZ_S"), ibooker);

  ibooker.setCurrentFolder(top_folder + "/Barrel/");
  numberClusters_Barrel_ =
      phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("NClusters_Barrel"), ibooker);

  //Now book layer wise histos
  edm::ESWatcher<TrackerDigiGeometryRecord> theTkDigiGeomWatcher;
  if (theTkDigiGeomWatcher.check(iSetup)) {
    for (auto const& det_u : tkGeom_->detUnits()) {
      //Always check TrackerNumberingBuilder before changing this part
      //continue if Pixel
      if ((det_u->subDetector() == GeomDetEnumerators::SubDetector::P2PXB ||
           det_u->subDetector() == GeomDetEnumerators::SubDetector::P2PXEC))
        continue;
      unsigned int detId_raw = det_u->geographicalId().rawId();
      edm::LogInfo("Phase2OTMonitorCluster")
          << "Detid:" << detId_raw << "\tsubdet=" << det_u->subDetector()
          << "\t key=" << phase2tkutil::getHistoId(detId_raw, tTopo_, 0.0, phase2tkutil::LAYER, false) << std::endl;
      bookLayerHistos(ibooker, detId_raw, top_folder);
    }
  }
}

//////////////////Layer Histo/////////////////////////////////
void Phase2OTMonitorCluster::bookLayerHistos(DQMStore::IBooker& ibooker, uint32_t det_id, std::string& subdir) {
  bool reducedOutput = config_.getParameter<bool>("ReducedOutput");
  for (int bookingDepth = phase2tkutil::WHOLE_TK; bookingDepth <= phase2tkutil::LAYER; bookingDepth++) {
    // Skip booking if barrel det and endcap-only depth
    if ((bookingDepth >= phase2tkutil::SHELL && bookingDepth < phase2tkutil::LAYER) &&
        DetId(det_id).subdetId() == SiStripSubdetector::TOB)
      continue;

    // skip intermediate depth level booking for reduced configurations
    if (reducedOutput && bookingDepth >= phase2tkutil::SHELL && bookingDepth != phase2tkutil::LAYER)
      continue;

    int key = phase2tkutil::getNumericHistoId(det_id, tTopo_, 0.0, bookingDepth);

    if (layerMEs_.find(key) == layerMEs_.end()) {
      std::string folderName = phase2tkutil::getHistoId(det_id, tTopo_, 0.0, bookingDepth, false);
      std::string prettyName = phase2tkutil::getHistoId(det_id, tTopo_, 0.0, bookingDepth, true);
      ibooker.cd();
      ibooker.setCurrentFolder(subdir + "/" + folderName);
      edm::LogInfo("Phase2OTMonitorCluster") << " Booking Histograms in: " << subdir + "/" + folderName;
      ClusterMEs local_mes;

      TrackerGeometry::ModuleType mType = tkGeom_->getDetectorType(det_id);
      if (mType == TrackerGeometry::ModuleType::Ph2PSP) {
        local_mes.nClusters_P = phase2tkutil::book1DFromPSet(
            config_.getParameter<edm::ParameterSet>("NClustersLayer_P"), ibooker, prettyName, bookingDepth);
        local_mes.ClusterSize_P =
            phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("ClusterSize_P"), ibooker, prettyName);
      }
      if (mType == TrackerGeometry::ModuleType::Ph2SS && bookingDepth == 6) {
        //Book the right number of histograms per layer
        unsigned int nLadders = 0;
        unsigned int nModules = 0;

        const auto theLayer = tTopo_->getOTLayerNumber(det_id);

        TrackerGeometry::DetIdContainer theDetIds = tkGeom_->detIds();
        for (auto detid : theDetIds) {
          // Only count 2S modules in the same layer as ref
          if (tkGeom_->getDetectorType(detid) != TrackerGeometry::ModuleType::Ph2SS)
            continue;
          if (tTopo_->getOTLayerNumber(detid) != theLayer)
            continue;

          const bool isBarrel = (detid.subdetId() == SiStripSubdetector::TOB);
          if (isBarrel) {
            nLadders = std::max(nLadders, tTopo_->tobRod(detid));
            nModules = std::max(nModules, tTopo_->module(detid));
          }
        }

        //Book the histograms
        local_mes.PositionOfClusters_2SLadder.resize(nLadders, nullptr);
        auto pos2SModulePSet = config_.getParameter<edm::ParameterSet>("PositionOfClusters_2S");
        if (pos2SModulePSet.getParameter<bool>("switch"))
          local_mes.PositionOfClusters_2S.resize(nModules, nullptr);
        auto xCorrelationModulePSet = config_.getParameter<edm::ParameterSet>("xCorrelationModule_2S");
        if (xCorrelationModulePSet.getParameter<bool>("switch")) {
          local_mes.ModuleXCorrelation_2S.resize(nModules, nullptr);
          local_mes.ClusterX_vs_sensor_Module.resize(nModules, {});
        }

        for (unsigned int ladderNum = 1; ladderNum <= nLadders; ladderNum++) {
          auto pos2SLadderPSet = config_.getParameter<edm::ParameterSet>("PositionOfClusters_2SLadder");
          pos2SLadderPSet.addParameter<std::string>("name", "Position_Of_Clusters_2S_Lad_" + std::to_string(ladderNum));
          pos2SLadderPSet.addParameter<std::string>(
              "title",
              "Position Of Clusters in 2S modules in {} ladder " + std::to_string(ladderNum) + ";Module;Half-module;");
          local_mes.PositionOfClusters_2SLadder[ladderNum - 1] =
              phase2tkutil::book2DFromPSet(pos2SLadderPSet, ibooker, prettyName);

          if (pos2SModulePSet.getParameter<bool>("switch")) {
            for (unsigned int moduleNum = 1; moduleNum <= nModules; moduleNum++) {
              pos2SModulePSet.addParameter<std::string>(
                  "name",
                  "Position_Of_Clusters_2S_Lad_" + std::to_string(ladderNum) + "_Mod_" + std::to_string(moduleNum));
              pos2SModulePSet.addParameter<std::string>("title",
                                                        "Position Of Clusters in 2S modules in {} ladder " +
                                                            std::to_string(ladderNum) + " module " +
                                                            std::to_string(moduleNum) + ";Strip;Half-module;");
              local_mes.PositionOfClusters_2S[moduleNum - 1] =
                  phase2tkutil::book2DFromPSet(pos2SModulePSet, ibooker, prettyName);

              xCorrelationModulePSet.addParameter<std::string>(
                  "name", "X_Correlation_Clusters_Module_" + std::to_string(moduleNum));
              xCorrelationModulePSet.addParameter<std::string>(
                  "title",
                  "Local X of clusters in {} present in top and bottom sensor of module " + std::to_string(moduleNum) +
                      ";top sensor X [cm];lower sensor X [cm]");
              local_mes.ModuleXCorrelation_2S[moduleNum - 1] =
                  phase2tkutil::book2DFromPSet(xCorrelationModulePSet, ibooker, prettyName);
            }
          }
        }
      }

      local_mes.nClusters_S = phase2tkutil::book1DFromPSet(
          config_.getParameter<edm::ParameterSet>("NClustersLayer_S"), ibooker, prettyName, bookingDepth);

      local_mes.ClusterSize_S =
          phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("ClusterSize_S"), ibooker, prettyName);

      layerMEs_.emplace(key, local_mes);
    }  //if block layerME find
  }
}

void Phase2OTMonitorCluster::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  phase2tkutil::getTemplatePSets(desc);
  // CRACK specific geometry
  phase2tkutil::add2DDesc(desc,
                          "CrackClusterSizeMap",
                          "Crack_ClusterSize",
                          "CRACK cluster size map",
                          "Module",
                          "Layer",
                          12,
                          0.5,
                          12.5,
                          6,
                          0.5,
                          6.5);
  // Works with many geometries,
  // but should be off by default.
  phase2tkutil::add2DDesc(desc,
                          "PositionOfClusters_2S",
                          "Position_Clusters_2S_module",
                          "Positions of clusters in 2S module",
                          "Strip",
                          "Half-module",
                          1016,
                          0.5,
                          1016.5,
                          5,
                          -2.5,
                          2.5);
  phase2tkutil::add2DDesc(
      desc, "xCorrelationModule_2S", "", "", "Cluster1 X", "Cluster2 X", 110, -5.5, 5.5, 110, -5.5, 5.5);

  phase2tkutil::add1DDesc(desc,
                          "NClusters_Barrel",
                          "Num_Clusters_Barrel",
                          "Number of clusters per Barrel Layer",
                          "Barrel Layer",
                          "Number of clusters",
                          6,
                          0.5,
                          6.5);
  phase2tkutil::add2DDesc(desc,
                          "GlobalPositionXY_P",
                          "Cluster_Global_Position_XY_P",
                          "Cluster Position XY P",
                          "Cluster position x [cm]",
                          "Cluster position y [cm]",
                          1250,
                          -125.0,
                          125.0,
                          1250,
                          -125.0,
                          125.0);
  phase2tkutil::add2DDesc(desc,
                          "GlobalPositionXY_S",
                          "Cluster_Global_Position_XY_S",
                          "Cluster Position XY S",
                          "Cluster position x [cm]",
                          "Cluster position y [cm]",
                          1250,
                          -125.0,
                          125.0,
                          1250,
                          -125.0,
                          125.0);
  phase2tkutil::add2DDesc(desc,
                          "GlobalPositionRZ_P",
                          "Cluster_Global_Position_RZ_P",
                          "Cluster Position RZ P",
                          "Cluster position z [cm]",
                          "Cluster position #rho [cm]",
                          1500,
                          -300.0,
                          300.0,
                          1250,
                          0.0,
                          125.0);
  phase2tkutil::add2DDesc(desc,
                          "GlobalPositionRZ_S",
                          "Cluster_Global_Position_RZ_S",
                          "Cluster Position RZ S",
                          "Cluster position z [cm]",
                          "Cluster position #rho [cm]",
                          1500,
                          -300.0,
                          300.0,
                          1250,
                          0.0,
                          125.0);

  //Layer wise histos
  phase2tkutil::add1DDesc(desc,
                          "NClustersLayer_P",
                          "Num_Clusters_Per_Event_P",
                          "Number Of Clusters in Pixels in {}",
                          "Number of clusters per event (macro pixel sensor)",
                          "",
                          150,
                          0.0,
                          300000);
  phase2tkutil::add1DDesc(desc,
                          "NClustersLayer_S",
                          "Num_Clusters_Per_Event_S",
                          "Number Of Clusters in strips in {}",
                          "Number of clusters per event (strip sensor)",
                          "",
                          150,
                          0.0,
                          300000);

  phase2tkutil::add1DDesc(desc,
                          "ClusterSize_P",
                          "Cluster_Size_P",
                          "Cluster Size in Pixels in {}",
                          "Cluster size (macro pixel sensor)",
                          "",
                          31,
                          -0.5,
                          30.5);
  phase2tkutil::add1DDesc(desc,
                          "ClusterSize_S",
                          "Cluster_Size_S",
                          "Cluster Size in strips in {}",
                          "Cluster size (strip sensor)",
                          "",
                          31,
                          -0.5,
                          30.5);

  phase2tkutil::add2DDesc(
      desc, "PositionOfClusters_2SLadder", "Position_Clusters_2S_Ladder", "", "", "", 25, -12.5, 12.5, 5, -2.5, 2.5);

  desc.add<std::string>("TopFolderName", "OuterTracker");
  desc.add<bool>("ReducedOutput", false);
  desc.add<edm::InputTag>("clusterSrc", edm::InputTag("siPhase2Clusters"));
  descriptions.add("Phase2OTMonitorCluster", desc);
}
DEFINE_FWK_MODULE(Phase2OTMonitorCluster);
