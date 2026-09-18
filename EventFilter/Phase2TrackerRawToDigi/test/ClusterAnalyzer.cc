// EDAnalyzer producing a small ntuple containing properties of Phase2TrackerCluster1D,
// to be used to debug the clusters-to-raw and raw-to-cluster steps

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "DataFormats/FEDRawData/interface/RawDataBuffer.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "CondFormats/SiPhase2TrackerObjects/interface/TrackerDetToDTCELinkCablingMap.h"
#include "CondFormats/SiPhase2TrackerObjects/interface/DTCELinkId.h"
#include "CondFormats/DataRecord/interface/TrackerDetToDTCELinkCablingMapRcd.h"

#include "TTree.h"
#include "TROOT.h"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

class ClusterAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  ClusterAnalyzer(const edm::ParameterSet& pset);
  ~ClusterAnalyzer() override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const& iEvent, edm::EventSetup const&) override {};
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  void beginJob() override;
  void endJob() override;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  const edm::ESGetToken<TrackerDetToDTCELinkCablingMap, TrackerDetToDTCELinkCablingMapRcd> cablingMapToken_;
  const edm::EDGetTokenT<Phase2TrackerCluster1DCollectionNew> token_;
  const TrackerTopology* tTopo_ = nullptr;
  const TrackerGeometry* tGeom_ = nullptr;
  const TrackerDetToDTCELinkCablingMap* cablingMap_ = nullptr;
  std::map<int, std::pair<int, int>> stackMap_;

  edm::Service<TFileService> fs_;
  TTree* outTree_;
  ofstream logfile_;

  std::vector<float> clusterR_;
  std::vector<float> clusterZ_;
  std::vector<unsigned int> clusterCol_;
  std::vector<unsigned int> clusterRow_;
  std::vector<uint32_t> detId_;
  std::vector<float> clusterCenter_;
  std::vector<int> clusterSize_;
  std::vector<float> clusterLocalX_;
  std::vector<float> clusterLocalY_;
  std::vector<float> clusterGlobalX_;
  std::vector<float> clusterGlobalY_;
  std::vector<float> clusterGlobalZ_;

  std::vector<bool> isPSModulePixel_;
  std::vector<bool> isPSModuleStrip_;
  std::vector<bool> is2SModule_;

  std::vector<int> dtcID_;
  unsigned long evt_n_;
};

ClusterAnalyzer::ClusterAnalyzer(const edm::ParameterSet& pset)
    : geomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>()),
      topoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd, edm::Transition::BeginRun>()),
      cablingMapToken_(
          esConsumes<TrackerDetToDTCELinkCablingMap, TrackerDetToDTCELinkCablingMapRcd, edm::Transition::BeginRun>()),
      token_(consumes<Phase2TrackerCluster1DCollectionNew>(pset.getParameter<edm::InputTag>("ProductLabel"))) {
  // Initialize the log file
  //   logfile_.open("ClusterAnalyzer_output.txt");
  //   if (!logfile_.is_open()) {
  //     throw cms::Exception("OutputFileError") << "Failed to open log file for writing.";
  //   }
}

ClusterAnalyzer::~ClusterAnalyzer() {
  // Close the log file
  //   logfile_.close();
}

void ClusterAnalyzer::beginJob() {
  outTree_ = fs_->make<TTree>("ClusterTree", "ClusterTree");

  outTree_->Branch("evt_n", &evt_n_, "evt_n/I");
  outTree_->Branch("detId", &detId_);
  outTree_->Branch("dtcID", &dtcID_);
  outTree_->Branch("isPSModulePixel", &isPSModulePixel_);
  outTree_->Branch("isPSModuleStrip", &isPSModuleStrip_);
  outTree_->Branch("is2SModule", &is2SModule_);
  outTree_->Branch("clusterCol", &clusterCol_);
  outTree_->Branch("clusterRow", &clusterRow_);
  outTree_->Branch("clusterR", &clusterR_);
  outTree_->Branch("clusterZ", &clusterZ_);
  outTree_->Branch("clusterCenter", &clusterCenter_);
  outTree_->Branch("clusterSize", &clusterSize_);
  outTree_->Branch("clusterLocalX", &clusterLocalX_);
  outTree_->Branch("clusterLocalY", &clusterLocalY_);
  outTree_->Branch("clusterGlobalX", &clusterGlobalX_);
  outTree_->Branch("clusterGlobalY", &clusterGlobalY_);
  outTree_->Branch("clusterGlobalZ", &clusterGlobalZ_);
}

void ClusterAnalyzer::endJob() {
  //     outTree_->GetDirectory()->cd();
  outTree_->Write();
}

void ClusterAnalyzer::beginRun(edm::Run const& run, edm::EventSetup const& es) {
  tGeom_ = &es.getData(geomToken_);
  tTopo_ = &es.getData(topoToken_);
  cablingMap_ = &es.getData(cablingMapToken_);
}

void ClusterAnalyzer::analyze(const edm::Event& event, const edm::EventSetup& es) {
  edm::Handle<Phase2TrackerCluster1DCollectionNew> clusters_handle;
  event.getByToken(token_, clusters_handle);

  evt_n_ = event.id().event();

  // Clear all vectors for this event
  detId_.clear();
  dtcID_.clear();
  isPSModulePixel_.clear();
  isPSModuleStrip_.clear();
  is2SModule_.clear();
  clusterCol_.clear();
  clusterRow_.clear();
  clusterR_.clear();
  clusterZ_.clear();
  clusterCenter_.clear();
  clusterSize_.clear();
  clusterLocalX_.clear();
  clusterLocalY_.clear();
  clusterGlobalX_.clear();
  clusterGlobalY_.clear();
  clusterGlobalZ_.clear();

  for (const auto& DSVItr : *clusters_handle) {
    uint32_t rawid(DSVItr.detId());
    DetId detId(rawid);
    const GeomDetUnit* geomDetUnit(tGeom_->idToDetUnit(detId));
    if (!geomDetUnit)
      continue;

    uint32_t current_detId = detId.rawId();
    bool current_isPSModulePixel = tGeom_->getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PSP;
    bool current_isPSModuleStrip = tGeom_->getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PSS;
    bool current_is2SModule = tGeom_->getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2SS;

    // Get dtcID for this module
    int current_dtcID = -1;
    if (cablingMap_->knowsDetId(current_detId - 1)) {
      auto equal_range = cablingMap_->detIdToDTCELinkId(current_detId - 1);
      for (auto it = equal_range.first; it != equal_range.second; ++it) {
        current_dtcID = it->second.dtc_id();
        break; // Take the first one
      }
    } else if (cablingMap_->knowsDetId(current_detId - 2)) {
      auto equal_range = cablingMap_->detIdToDTCELinkId(current_detId - 2);
      for (auto it = equal_range.first; it != equal_range.second; ++it) {
        current_dtcID = it->second.dtc_id();
        break;
      }
    }

    // Loop over clusters in this module
    for (const auto& clusterItr : DSVItr) {
      // Store module-level info (repeated for each cluster in this module)
      detId_.push_back(current_detId);
      dtcID_.push_back(current_dtcID);
      isPSModulePixel_.push_back(current_isPSModulePixel);
      isPSModuleStrip_.push_back(current_isPSModuleStrip);
      is2SModule_.push_back(current_is2SModule);
      
      // Store cluster-specific info
      clusterCenter_.push_back(clusterItr.center());
      clusterSize_.push_back(clusterItr.size());
      clusterCol_.push_back(clusterItr.column());
      clusterRow_.push_back(clusterItr.firstRow());

      MeasurementPoint mpCluster(clusterItr.center(), clusterItr.column() + 0.5);
      Local3DPoint localPosCluster = geomDetUnit->topology().localPosition(mpCluster);
      Global3DPoint globalPosCluster = geomDetUnit->surface().toGlobal(localPosCluster);

      clusterLocalX_.push_back(localPosCluster.x());
      clusterLocalY_.push_back(localPosCluster.y());

      clusterGlobalX_.push_back(globalPosCluster.x());
      clusterGlobalY_.push_back(globalPosCluster.y());
      clusterGlobalZ_.push_back(globalPosCluster.z());

      clusterR_.push_back(globalPosCluster.perp());
      clusterZ_.push_back(globalPosCluster.z());

    }
  }
  
  // Fill the tree with one entry per event (containing all clusters as vectors)
  outTree_->Fill();
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(ClusterAnalyzer);
