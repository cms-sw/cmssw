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
//
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
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
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
    MonitorElement* XYGlobalPositionMap_P = nullptr;
    MonitorElement* XYLocalPositionMap_P = nullptr;

    MonitorElement* nClusters_S = nullptr;
    MonitorElement* ClusterSize_S = nullptr;
    MonitorElement* XYGlobalPositionMap_S = nullptr;
    MonitorElement* XYLocalPositionMap_S = nullptr;

    std::vector<MonitorElement*> PositionOfClusters_2S;
    std::vector<MonitorElement*> PositionOfClusters_2SLadder;
  };
  MonitorElement* numberClusters_;
  MonitorElement* globalXY_P_;
  MonitorElement* globalRZ_P_;
  MonitorElement* globalXY_S_;
  MonitorElement* globalRZ_S_;
  MonitorElement* numberClusters_Barrel_;
  MonitorElement* crackOverview_;

  void fillOTHistos(const edm::Event& iEvent);

  void bookLayerHistos(DQMStore::IBooker& ibooker, uint32_t det_it, std::string& subdir);

  std::map<std::string, ClusterMEs> layerMEs_;

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

  // Number of clusters
  std::map<std::string, unsigned int> nClustersCounter_P;  //map of detidkey vs #cls
  std::map<std::string, unsigned int> nClustersCounter_S;  //map of detidkey vs #cls
  unsigned int nclus = 0;                                  //global counter
  for (const auto& DSVItr : *clusterHandle) {
    // Getting the id of detector unit
    uint32_t rawid(DSVItr.detId());
    DetId detId(rawid);
    const GeomDetUnit* geomDetUnit(tkGeom_->idToDetUnit(detId));
    if (!geomDetUnit)
      continue;

    std::string folderkey = phase2tkutil::getOTHistoId(detId, tTopo_);

    TrackerGeometry::ModuleType mType = tkGeom_->getDetectorType(detId);
    // initialize the nhit counters if they don't exist for this layer
    //the check on the detId is needed to avoid checking at the filling stage
    if (mType == TrackerGeometry::ModuleType::Ph2PSP) {
      auto counterDet = nClustersCounter_P.find(folderkey);
      if (counterDet == nClustersCounter_P.end())
        nClustersCounter_P.emplace(folderkey, DSVItr.size());
      else
        counterDet->second += DSVItr.size();
    } else if (mType == TrackerGeometry::ModuleType::Ph2PSS || mType == TrackerGeometry::ModuleType::Ph2SS) {
      auto counterDet = nClustersCounter_S.find(folderkey);
      if (counterDet == nClustersCounter_S.end())
        nClustersCounter_S.emplace(folderkey, DSVItr.size());
      else
        counterDet->second += DSVItr.size();
    }
    nclus += DSVItr.size();

    for (const auto& clusterItr : DSVItr) {
      MeasurementPoint mpCluster(clusterItr.center(), clusterItr.column() + 0.5);
      Local3DPoint localPosCluster = geomDetUnit->topology().localPosition(mpCluster);
      Global3DPoint globalPosCluster = geomDetUnit->surface().toGlobal(localPosCluster);
      double gx = globalPosCluster.x();
      double gy = globalPosCluster.y();
      double gz = globalPosCluster.z();
      double gr = globalPosCluster.perp();
      auto layerMEit = layerMEs_.find(folderkey);
      if (layerMEit == layerMEs_.end())
        continue;
      ClusterMEs& local_mes = layerMEit->second;
      if (detId.subdetId() == SiStripSubdetector::TOB) {
        numberClusters_Barrel_->Fill(tTopo_->layer(detId));
      }
      if (mType == TrackerGeometry::ModuleType::Ph2PSP) {
        globalXY_P_->Fill(gx, gy);
        globalRZ_P_->Fill(gz, gr);
        local_mes.ClusterSize_P->Fill(clusterItr.size());
        local_mes.XYLocalPositionMap_P->Fill(localPosCluster.x(), localPosCluster.y());

        if (local_mes.XYGlobalPositionMap_P != nullptr)  //make this optional
          local_mes.XYGlobalPositionMap_P->Fill(gx, gy);
      } else if (mType == TrackerGeometry::ModuleType::Ph2PSS || mType == TrackerGeometry::ModuleType::Ph2SS) {
        globalXY_S_->Fill(gx, gy);
        globalRZ_S_->Fill(gz, gr);
        local_mes.ClusterSize_S->Fill(clusterItr.size());
        local_mes.XYLocalPositionMap_S->Fill(localPosCluster.x(), localPosCluster.y());

        if (local_mes.XYGlobalPositionMap_S != nullptr)  //make this optional
          local_mes.XYGlobalPositionMap_S->Fill(gx, gy);
      }
      if (mType == TrackerGeometry::ModuleType::Ph2SS) {
        if (detId.subdetId() == SiStripSubdetector::TOB) {
          unsigned int module = tTopo_->module(rawid);
          //If column is on the bottom of the sensor, *-1 to distinguish it from top
          int topOrBottomColumn = (tTopo_->isLower(rawid) ? (clusterItr.column() + 1) * -1 : (clusterItr.column() + 1));
          if (module < local_mes.PositionOfClusters_2S.size() && local_mes.PositionOfClusters_2S[module]) {
            local_mes.PositionOfClusters_2S[module]->Fill(clusterItr.center(), topOrBottomColumn);
          }
          unsigned int ladder = tTopo_->tobRod(rawid);
          if (local_mes.PositionOfClusters_2SLadder[ladder] != nullptr) {
            //Adapt module numbers from 1 to 24 into -12 to +12
            int signedModule = module <= 12 ? module : -(module - 12);
            local_mes.PositionOfClusters_2SLadder[ladder]->Fill(signedModule, topOrBottomColumn);
          }
          if (crackOverview_)
            crackOverview_->Fill(module, tTopo_->getOTLayerNumber(rawid) - 0.05 + (module % 2 * 0.1));
        }
      }
    }
  }
  for (const auto& it : nClustersCounter_P) {
    if (layerMEs_.find(it.first) == layerMEs_.end())
      continue;
    if (layerMEs_[it.first].nClusters_P != nullptr)  //this check should not be required though
      layerMEs_[it.first].nClusters_P->Fill(it.second);
  }
  for (const auto& it : nClustersCounter_S) {
    if (layerMEs_.find(it.first) == layerMEs_.end())
      continue;
    if (layerMEs_[it.first].nClusters_S != nullptr)  //this check should not be required though
      layerMEs_[it.first].nClusters_S->Fill(it.second);
  }
  numberClusters_->Fill(nclus);
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

  numberClusters_ = phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("GlobalNClusters"), ibooker);

  globalXY_P_ = phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("GlobalPositionXY_P"), ibooker);

  globalRZ_P_ = phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("GlobalPositionRZ_P"), ibooker);

  globalXY_S_ = phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("GlobalPositionXY_S"), ibooker);

  globalRZ_S_ = phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("GlobalPositionRZ_S"), ibooker);

  numberClusters_Barrel_ =
      phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("NClusters_Barrel"), ibooker);

  edm::ParameterSet Parameters = config_.getParameter<edm::ParameterSet>("CrackOverview");
  if (Parameters.getParameter<bool>("switch")) {
    crackOverview_ = ibooker.book2DPoly(Parameters.getParameter<std::string>("name"),
                                        Parameters.getParameter<std::string>("title"),
                                        Parameters.getParameter<double>("xmin"),
                                        Parameters.getParameter<double>("xmax"),
                                        Parameters.getParameter<double>("ymin"),
                                        Parameters.getParameter<double>("ymax"));
    if (crackOverview_->getTH2Poly()->GetNumberOfBins() == 0) {
      double yOffset = 0;
      for (int layer = 1; layer < 7; layer++) {
        for (int module = 1; module < 13; module++) {
          if (module % 2 == 0)
            yOffset = -0.1;
          else
            yOffset = 0;
          crackOverview_->addBin(module - 0.7, layer + yOffset, module + 0.7, layer + yOffset + 0.1);
        }
      }
    }
    crackOverview_->getTH2Poly()->SetStats(false);
    crackOverview_->setOption("z0");

  } else
    crackOverview_ = nullptr;

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
      edm::LogInfo("Phase2ITMonitorRecHit") << "Detid:" << detId_raw << "\tsubdet=" << det_u->subDetector()
                                            << "\t key=" << phase2tkutil::getITHistoId(detId_raw, tTopo_) << std::endl;
      bookLayerHistos(ibooker, detId_raw, top_folder);
    }
  }
}

//////////////////Layer Histo/////////////////////////////////
void Phase2OTMonitorCluster::bookLayerHistos(DQMStore::IBooker& ibooker, uint32_t det_id, std::string& subdir) {
  std::string folderName = phase2tkutil::getOTHistoId(det_id, tTopo_);
  if (folderName.empty()) {
    edm::LogWarning("Phase2OTMonitorCluster") << ">>>> Invalid histo_id ";
    return;
  }
  if (layerMEs_.find(folderName) == layerMEs_.end()) {
    ibooker.cd();
    ibooker.setCurrentFolder(subdir + "/" + folderName);
    edm::LogInfo("Phase2OTMonitorCluster") << " Booking Histograms in: " << subdir + "/" + folderName;
    ClusterMEs local_mes;
    TrackerGeometry::ModuleType mType = tkGeom_->getDetectorType(det_id);
    if (mType == TrackerGeometry::ModuleType::Ph2PSP) {
      local_mes.nClusters_P =
          phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("NClustersLayer_P"), ibooker);
      local_mes.ClusterSize_P =
          phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("ClusterSize_P"), ibooker);
      local_mes.XYGlobalPositionMap_P =
          phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("GlobalPositionXY_perlayer_P"), ibooker);
      local_mes.XYLocalPositionMap_P =
          phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("LocalPositionXY_P"), ibooker);
    }
    if (mType == TrackerGeometry::ModuleType::Ph2SS) {
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
      local_mes.PositionOfClusters_2SLadder.resize(nLadders + 1, nullptr);
      auto pos2SModulePSet = config_.getParameter<edm::ParameterSet>("PositionOfClusters_2S");
      if (pos2SModulePSet.getParameter<bool>("switch"))
        local_mes.PositionOfClusters_2S.resize(nModules + 1, nullptr);

      for (unsigned int ladderNum = 1; ladderNum <= nLadders; ladderNum++) {
        auto pos2SLadderPSet = config_.getParameter<edm::ParameterSet>("PositionOfClusters_2SLadder");
        pos2SLadderPSet.addParameter<std::string>(
            "name", "PositionOfClusters_2S_Lay" + std::to_string(theLayer) + "_Lad" + std::to_string(ladderNum));
        pos2SLadderPSet.addParameter<std::string>("title",
                                                  "PositionOfClusters_2S_Lay" + std::to_string(theLayer) + "_Lad" +
                                                      std::to_string(ladderNum) + ";Module;Half-module;");
        local_mes.PositionOfClusters_2SLadder[ladderNum] = phase2tkutil::book2DFromPSet(pos2SLadderPSet, ibooker);
        if (local_mes.PositionOfClusters_2SLadder[ladderNum] != nullptr) {
          local_mes.PositionOfClusters_2SLadder[ladderNum]->getTH2F()->SetStats(false);
          local_mes.PositionOfClusters_2SLadder[ladderNum]->setOption("z");
        }
        if (pos2SModulePSet.getParameter<bool>("switch")) {
          for (unsigned int moduleNum = 1; moduleNum <= nModules; moduleNum++) {
            auto pos2SModulePSet = config_.getParameter<edm::ParameterSet>("PositionOfClusters_2S");
            pos2SModulePSet.addParameter<std::string>("name",
                                                      "PositionOfClusters_2S_Lay" + std::to_string(theLayer) + "_Lad" +
                                                          std::to_string(ladderNum) + "_Mod" +
                                                          std::to_string(moduleNum));
            pos2SModulePSet.addParameter<std::string>("title",
                                                      "PositionOfClusters_2S_Lay" + std::to_string(theLayer) + "_Lad" +
                                                          std::to_string(ladderNum) + "_Mod" +
                                                          std::to_string(moduleNum) + ";Strip;Half-module;");
            local_mes.PositionOfClusters_2S[moduleNum] = phase2tkutil::book2DFromPSet(pos2SModulePSet, ibooker);
            if (local_mes.PositionOfClusters_2S[moduleNum] != nullptr) {
              local_mes.PositionOfClusters_2S[moduleNum]->getTH2F()->SetStats(false);
              local_mes.PositionOfClusters_2S[moduleNum]->setOption("z");
            }
          }
        }
      }
    }

    local_mes.nClusters_S =
        phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("NClustersLayer_S"), ibooker);

    local_mes.ClusterSize_S =
        phase2tkutil::book1DFromPSet(config_.getParameter<edm::ParameterSet>("ClusterSize_S"), ibooker);

    local_mes.XYGlobalPositionMap_S =
        phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("GlobalPositionXY_perlayer_S"), ibooker);

    local_mes.XYLocalPositionMap_S =
        phase2tkutil::book2DFromPSet(config_.getParameter<edm::ParameterSet>("LocalPositionXY_S"), ibooker);

    layerMEs_.emplace(folderName, local_mes);
  }  //if block layerME find
}

void Phase2OTMonitorCluster::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // rechitMonitorOT
  edm::ParameterSetDescription desc;
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "NumberOfClusters");
    psd0.add<std::string>("title", "Number_Of_Clusters;Number of clusters per event;");
    psd0.add<double>("xmin", 0.0);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 350000.0);
    psd0.add<int>("NxBins", 150);
    desc.add<edm::ParameterSetDescription>("GlobalNClusters", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Crack_Overview");
    psd0.add<std::string>("title", "Crack_Overview_clusters;Module;Layer");
    psd0.add<double>("xmin", 0.0);
    psd0.add<bool>("switch", false);
    psd0.add<double>("xmax", 13.0);
    psd0.add<double>("ymin", 0.0);
    psd0.add<double>("ymax", 7.5);
    desc.add<edm::ParameterSetDescription>("CrackOverview", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Cluster_Position_XY_P");
    psd0.add<std::string>("title", "Cluster_Position_XY_P;Cluster position x [cm];Cluster position y [cm];");
    psd0.add<int>("NxBins", 1250);
    psd0.add<double>("xmin", -125.0);
    psd0.add<double>("xmax", 125.0);
    psd0.add<int>("NyBins", 1250);
    psd0.add<double>("ymin", -125.0);
    psd0.add<double>("ymax", 125.0);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("GlobalPositionXY_P", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Cluster_Position_XY_S");
    psd0.add<std::string>("title", "Cluster_Position_XY_S;Cluster position x [cm];Cluster position y [cm];");
    psd0.add<int>("NxBins", 1250);
    psd0.add<double>("xmin", -125.0);
    psd0.add<double>("xmax", 125.0);
    psd0.add<int>("NyBins", 1250);
    psd0.add<double>("ymin", -125.0);
    psd0.add<double>("ymax", 125.0);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("GlobalPositionXY_S", psd0);
  }

  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Cluster_Position_RZ_P");
    psd0.add<std::string>("title", "Cluster_Position_RZ_P;Cluster position z [cm];Cluster position #rho [cm]");
    psd0.add<int>("NxBins", 1500);
    psd0.add<double>("xmin", -300.0);
    psd0.add<double>("xmax", 300.0);
    psd0.add<int>("NyBins", 1250);
    psd0.add<double>("ymin", 0.0);
    psd0.add<double>("ymax", 125.0);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("GlobalPositionRZ_P", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Cluster_Position_RZ_S");
    psd0.add<std::string>("title", "Cluster_Position_RZ_S;Cluster position z [cm];Cluster position #rho [cm]");
    psd0.add<int>("NxBins", 1500);
    psd0.add<double>("xmin", -300.0);
    psd0.add<double>("xmax", 300.0);
    psd0.add<int>("NyBins", 1250);
    psd0.add<double>("ymin", 0.0);
    psd0.add<double>("ymax", 125.0);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("GlobalPositionRZ_S", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "NumberOfClusters_Barrel");
    psd0.add<std::string>("title", "Number of clusters per Barrel Layer;Barrel Layer;Number of clusters");
    psd0.add<int>("NxBins", 7);
    psd0.add<double>("xmin", 0.5);
    psd0.add<double>("xmax", 7.5);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("NClusters_Barrel", psd0);
  }
  //Layer wise parameter
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "NumberOfClusters_Layer_P");
    psd0.add<std::string>("title", "Number_Of_Clusters_P_Layer;Number of clusters per event (macro pixel sensor);");
    psd0.add<double>("xmin", 0.0);
    psd0.add<double>("xmax", 28000.0);
    psd0.add<int>("NxBins", 150);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("NClustersLayer_P", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "NumberOfClusters_Layer_S");
    psd0.add<std::string>("title", "Number_Of_Clusters_S_Layer;Number of clusters per event (strip sensor);");
    psd0.add<double>("xmin", 0.0);
    psd0.add<double>("xmax", 28000.0);
    psd0.add<int>("NxBins", 150);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("NClustersLayer_S", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Cluster_Size_P");
    psd0.add<std::string>("title", "Cluster_Size_P;Cluster size (macro pixel sensor);");
    psd0.add<double>("xmin", -0.5);
    psd0.add<double>("xmax", 30.5);
    psd0.add<int>("NxBins", 31);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("ClusterSize_P", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Cluster_Size_S");
    psd0.add<std::string>("title", "Cluster_Size_S;Cluster size (strip sensor);");
    psd0.add<double>("xmin", -0.5);
    psd0.add<double>("xmax", 30.5);
    psd0.add<int>("NxBins", 31);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("ClusterSize_S", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Cluster_Position_XY_perLayer_P");
    psd0.add<std::string>("title", "Cluster_Position_XY_perLayer_P;Cluster position x [cm];Cluster position y [cm];");
    psd0.add<int>("NxBins", 1250);
    psd0.add<double>("xmin", -125.0);
    psd0.add<double>("xmax", 125.0);
    psd0.add<int>("NyBins", 1250);
    psd0.add<double>("ymin", -125.0);
    psd0.add<double>("ymax", 125.0);
    psd0.add<bool>("switch", false);
    desc.add<edm::ParameterSetDescription>("GlobalPositionXY_perlayer_P", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Cluster_Position_XY_perLayer_S");
    psd0.add<std::string>("title", "Cluster_Position_XY_perLayer_S;Cluster position x [cm];Cluster position y [cm];");
    psd0.add<int>("NxBins", 1250);
    psd0.add<double>("xmin", -125.0);
    psd0.add<double>("xmax", 125.0);
    psd0.add<int>("NyBins", 1250);
    psd0.add<double>("ymin", -125.0);
    psd0.add<double>("ymax", 125.0);
    psd0.add<bool>("switch", false);
    desc.add<edm::ParameterSetDescription>("GlobalPositionXY_perlayer_S", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Cluster_Local_Position_XY_P");
    psd0.add<std::string>("title", "Cluster_Local_Position_XY_P;Cluster position x [cm];Cluster position y [cm];");
    psd0.add<int>("NxBins", 25);
    psd0.add<double>("xmin", -5.0);
    psd0.add<double>("xmax", 5.0);
    psd0.add<int>("NyBins", 15);
    psd0.add<double>("ymin", -3.0);
    psd0.add<double>("ymax", 3.0);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("LocalPositionXY_P", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Cluster_Local_Position_XY_S");
    psd0.add<std::string>("title", "Cluster_Local_Position_XY_S;Cluster position x [cm];Cluster position y [cm];");
    psd0.add<int>("NxBins", 25);
    psd0.add<double>("xmin", -5.0);
    psd0.add<double>("xmax", 5.0);
    psd0.add<int>("NyBins", 15);
    psd0.add<double>("ymin", -3.0);
    psd0.add<double>("ymax", 3.0);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("LocalPositionXY_S", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "PositionOfClusters_2S_module");
    psd0.add<std::string>("title", "PositionsOfClusters_2S_module;Strip;Half-module;");
    psd0.add<int>("NxBins", 1016);
    psd0.add<double>("xmin", 0.5);
    psd0.add<double>("xmax", 1016.5);
    psd0.add<int>("NyBins", 5);
    psd0.add<double>("ymin", -2.5);
    psd0.add<double>("ymax", 2.5);
    psd0.add<bool>("switch", false);
    desc.add<edm::ParameterSetDescription>("PositionOfClusters_2S", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "PositionOfClusters_2S_Ladder");
    psd0.add<std::string>("title", "PositionsOfClusters_2S_Ladder;Module;Half-module;");
    psd0.add<int>("NxBins", 25);
    psd0.add<double>("xmin", -12.5);
    psd0.add<double>("xmax", 12.5);
    psd0.add<int>("NyBins", 5);
    psd0.add<double>("ymin", -2.5);
    psd0.add<double>("ymax", 2.5);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("PositionOfClusters_2SLadder", psd0);
  }

  desc.add<std::string>("TopFolderName", "TrackerPhase2OTCluster");
  desc.add<edm::InputTag>("clusterSrc", edm::InputTag("siPhase2Clusters"));
  descriptions.add("Phase2OTMonitorCluster", desc);
}
DEFINE_FWK_MODULE(Phase2OTMonitorCluster);
