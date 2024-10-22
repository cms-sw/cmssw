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
  };
  MonitorElement* numberClusters_;
  MonitorElement* globalXY_P_;
  MonitorElement* globalRZ_P_;
  MonitorElement* globalXY_S_;
  MonitorElement* globalRZ_S_;

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
      double gx = globalPosCluster.x() * 10.;
      double gy = globalPosCluster.y() * 10.;
      double gz = globalPosCluster.z() * 10.;
      double gr = globalPosCluster.perp() * 10.;
      auto layerMEit = layerMEs_.find(folderkey);
      if (layerMEit == layerMEs_.end())
        continue;
      ClusterMEs& local_mes = layerMEit->second;
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
    psd0.add<std::string>("title", ";Number of clusters per event;");
    psd0.add<double>("xmin", 0.0);
    psd0.add<bool>("switch", true);
    psd0.add<double>("xmax", 350000.0);
    psd0.add<int>("NxBins", 150);
    desc.add<edm::ParameterSetDescription>("GlobalNClusters", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Global_ClusterPosition_XY_P");
    psd0.add<std::string>("title", "Global_ClusterPosition_XY_P;x [mm];y [mm];");
    psd0.add<int>("NxBins", 1250);
    psd0.add<double>("xmin", -1250.0);
    psd0.add<double>("xmax", 1250.0);
    psd0.add<int>("NyBins", 1250);
    psd0.add<double>("ymin", -1250.0);
    psd0.add<double>("ymax", 1250.0);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("GlobalPositionXY_P", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Global_ClusterPosition_XY_S");
    psd0.add<std::string>("title", "Global_ClusterPosition_XY_S;x [mm];y [mm];");
    psd0.add<int>("NxBins", 1250);
    psd0.add<double>("xmin", -1250.0);
    psd0.add<double>("xmax", 1250.0);
    psd0.add<int>("NyBins", 1250);
    psd0.add<double>("ymin", -1250.0);
    psd0.add<double>("ymax", 1250.0);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("GlobalPositionXY_S", psd0);
  }

  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Global_ClusterPosition_RZ_P");
    psd0.add<std::string>("title", "Global_ClusterPosition_RZ_P;z [mm];r [mm]");
    psd0.add<int>("NxBins", 1500);
    psd0.add<double>("xmin", -3000.0);
    psd0.add<double>("xmax", 3000.0);
    psd0.add<int>("NyBins", 1250);
    psd0.add<double>("ymin", 0.0);
    psd0.add<double>("ymax", 1250.0);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("GlobalPositionRZ_P", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "Global_ClusterPosition_RZ_S");
    psd0.add<std::string>("title", "Global_ClusterPosition_RZ_S;z [mm];r [mm]");
    psd0.add<int>("NxBins", 1500);
    psd0.add<double>("xmin", -3000.0);
    psd0.add<double>("xmax", 3000.0);
    psd0.add<int>("NyBins", 1250);
    psd0.add<double>("ymin", 0.0);
    psd0.add<double>("ymax", 1250.0);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("GlobalPositionRZ_S", psd0);
  }
  //Layer wise parameter
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "NumberOfClustersLayerP");
    psd0.add<std::string>("title", ";Number of clusters per event(macro pixel sensor);");
    psd0.add<double>("xmin", 0.0);
    psd0.add<double>("xmax", 28000.0);
    psd0.add<int>("NxBins", 150);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("NClustersLayer_P", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "NumberOfClustersLayerS");
    psd0.add<std::string>("title", ";Number of clusters per event(strip sensor);");
    psd0.add<double>("xmin", 0.0);
    psd0.add<double>("xmax", 28000.0);
    psd0.add<int>("NxBins", 150);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("NClustersLayer_S", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "ClusterSize_P");
    psd0.add<std::string>("title", ";cluster size(macro pixel sensor);");
    psd0.add<double>("xmin", -0.5);
    psd0.add<double>("xmax", 30.5);
    psd0.add<int>("NxBins", 31);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("ClusterSize_P", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "ClusterSize_S");
    psd0.add<std::string>("title", ";cluster size(strip sensor);");
    psd0.add<double>("xmin", -0.5);
    psd0.add<double>("xmax", 30.5);
    psd0.add<int>("NxBins", 31);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("ClusterSize_S", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "GlobalPositionXY_perlayer_P");
    psd0.add<std::string>("title", "GlobalClusterPositionXY_perlayer_P;x [mm];y [mm];");
    psd0.add<int>("NxBins", 1250);
    psd0.add<double>("xmin", -1250.0);
    psd0.add<double>("xmax", 1250.0);
    psd0.add<int>("NyBins", 1250);
    psd0.add<double>("ymin", -1250.0);
    psd0.add<double>("ymax", 1250.0);
    psd0.add<bool>("switch", false);
    desc.add<edm::ParameterSetDescription>("GlobalPositionXY_perlayer_P", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "GlobalPositionXY_perlayer_S");
    psd0.add<std::string>("title", "GlobalClusterPositionXY_perlayer_S;x [mm];y [mm];");
    psd0.add<int>("NxBins", 1250);
    psd0.add<double>("xmin", -1250.0);
    psd0.add<double>("xmax", 1250.0);
    psd0.add<int>("NyBins", 1250);
    psd0.add<double>("ymin", -1250.0);
    psd0.add<double>("ymax", 1250.0);
    psd0.add<bool>("switch", false);
    desc.add<edm::ParameterSetDescription>("GlobalPositionXY_perlayer_S", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "LocalPositionXY_P");
    psd0.add<std::string>("title", "LocalPositionXY_P;x ;y ;");
    psd0.add<int>("NxBins", 50);
    psd0.add<double>("xmin", -10.0);
    psd0.add<double>("xmax", 10.0);
    psd0.add<int>("NyBins", 50);
    psd0.add<double>("ymin", -10.0);
    psd0.add<double>("ymax", 10.0);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("LocalPositionXY_P", psd0);
  }
  {
    edm::ParameterSetDescription psd0;
    psd0.add<std::string>("name", "LocalPositionXY_S");
    psd0.add<std::string>("title", "LocalPositionXY_S;x ;y ;");
    psd0.add<int>("NxBins", 50);
    psd0.add<double>("xmin", -10.0);
    psd0.add<double>("xmax", 10.0);
    psd0.add<int>("NyBins", 50);
    psd0.add<double>("ymin", -10.0);
    psd0.add<double>("ymax", 10.0);
    psd0.add<bool>("switch", true);
    desc.add<edm::ParameterSetDescription>("LocalPositionXY_S", psd0);
  }

  desc.add<std::string>("TopFolderName", "TrackerPhase2OTCluster");
  desc.add<edm::InputTag>("clusterSrc", edm::InputTag("siPhase2Clusters"));
  descriptions.add("Phase2OTMonitorCluster", desc);
}
DEFINE_FWK_MODULE(Phase2OTMonitorCluster);
