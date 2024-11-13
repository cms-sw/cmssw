#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetIdCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
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

#include "TTree.h"
#include "TROOT.h"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

class Phase2TrackerDumpDigi : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  Phase2TrackerDumpDigi(const edm::ParameterSet& pset);
  ~Phase2TrackerDumpDigi() override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void endRun(edm::Run const& iEvent, edm::EventSetup const&) override {};
  void analyze(const edm::Event&, const edm::EventSetup&);

private:
  virtual void beginJob();
  virtual void endJob();
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  const edm::EDGetTokenT<Phase2TrackerCluster1DCollectionNew> token_;
  const TrackerTopology* tTopo_ = nullptr;
  const TrackerGeometry* tGeom_ = nullptr;
  std::map<int, std::pair<int, int>> stackMap_;

  edm::Service<TFileService> fs_;
  TTree* outTree_;
  ofstream logfile_;
  
  float clusterR_;
  float clusterZ_;
  uint32_t detId_;
  float clusterCenter_;
  int clusterSize_;
  float clusterLocalX_;
  float clusterLocalY_;
  float clusterGlobalX_;
  float clusterGlobalY_;
  float clusterGlobalZ_;

   bool isPSModulePixel_ ;
   bool isPSModuleStrip_ ;
   bool is2SModule_      ;
  
};

Phase2TrackerDumpDigi::Phase2TrackerDumpDigi(const edm::ParameterSet& pset)
    : geomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>()),
      topoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd, edm::Transition::BeginRun>()),
      token_(consumes<Phase2TrackerCluster1DCollectionNew>(pset.getParameter<edm::InputTag>("ProductLabel"))) {

    // Initialize the log file
    logfile_.open("Phase2TrackerDumpDigi_output.txt");
//     logfile_.open("Phase2TrackerDumpDigi_original_output.txt");
    if (!logfile_.is_open()) {
        throw cms::Exception("OutputFileError") << "Failed to open log file for writing.";
    }
}

Phase2TrackerDumpDigi::~Phase2TrackerDumpDigi() {

    // Close the log file
    logfile_.close();
}

void Phase2TrackerDumpDigi::beginJob ()
{
    outTree_ = fs_->make<TTree>("ClusterTree","ClusterTree");

    outTree_->Branch("detId", &detId_, "detId/i");
    outTree_->Branch("isPSModulePixel", &isPSModulePixel_, "isPSModulePixel/O");
    outTree_->Branch("isPSModuleStrip", &isPSModuleStrip_, "isPSModuleStrip/O");
    outTree_->Branch("is2SModule", &is2SModule_, "is2SModule/O");
    outTree_->Branch("clusterR", &clusterR_, "clusterR/F");
    outTree_->Branch("clusterZ", &clusterZ_, "clusterZ/F");
    outTree_->Branch("clusterCenter", &clusterCenter_, "clusterCenter/F");
    outTree_->Branch("clusterSize", &clusterSize_, "clusterSize/I");
    outTree_->Branch("clusterLocalX", &clusterLocalX_, "clusterLocalX/F");
    outTree_->Branch("clusterLocalY", &clusterLocalY_, "clusterLocalY/F");
    outTree_->Branch("clusterGlobalX", &clusterGlobalX_, "clusterGlobalX/F");
    outTree_->Branch("clusterGlobalY", &clusterGlobalY_, "clusterGlobalY/F");
    outTree_->Branch("clusterGlobalZ", &clusterGlobalZ_, "clusterGlobalZ/F");

}
void Phase2TrackerDumpDigi::endJob ()
{
//     outTree_->GetDirectory()->cd();
    outTree_->Write();
}


void Phase2TrackerDumpDigi::beginRun(edm::Run const& run, edm::EventSetup const& es) {
    tGeom_ = &es.getData(geomToken_);
    tTopo_ = &es.getData(topoToken_);
}

void Phase2TrackerDumpDigi::analyze(const edm::Event& event, const edm::EventSetup& es) {
  edm::Handle<Phase2TrackerCluster1DCollectionNew> clusters_handle;
  event.getByToken(token_, clusters_handle);

  std::stringstream output;
  output << "size of clusters: " << clusters_handle.product()->size() << std::endl;

  for (const auto& DSVItr : *clusters_handle) {
//     output << "inside loop" << std::endl;

    uint32_t rawid(DSVItr.detId());
    DetId detId(rawid);
    const GeomDetUnit* geomDetUnit(tGeom_->idToDetUnit(detId));
    if (!geomDetUnit)
      continue;

    detId_ = detId.rawId();
    isPSModulePixel_ = tGeom_->getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PSP;
    isPSModuleStrip_ = tGeom_->getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PSS;
    is2SModule_      = tGeom_->getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2SS;
    output << "detId: " << detId.rawId() << "  " ;
    output << (isPSModulePixel_ ? "isPSModulePixel_" : (isPSModuleStrip_ ? "isPSModuleStrip_" : "is2SModule_")); 
    output << std::endl;
    
    for (const auto& clusterItr : DSVItr) {
      clusterCenter_ = clusterItr.center();
      clusterSize_ = clusterItr.size();

      MeasurementPoint mpCluster(clusterItr.center(), clusterItr.column() + 0.5);
      Local3DPoint localPosCluster = geomDetUnit->topology().localPosition(mpCluster);
      Global3DPoint globalPosCluster = geomDetUnit->surface().toGlobal(localPosCluster);

      clusterLocalX_ = localPosCluster.x();
      clusterLocalY_ = localPosCluster.y();
      
      clusterGlobalX_ = globalPosCluster.x();
      clusterGlobalY_ = globalPosCluster.y();
      clusterGlobalZ_ = globalPosCluster.z();

      clusterR_ = globalPosCluster.perp();
      clusterZ_ = globalPosCluster.z();

      output << "\t cluster r position: " << globalPosCluster.perp() << std::endl;
      output << "\t cluster global z position: " << globalPosCluster.z() << std::endl;

      outTree_->Fill();  // Fill the tree with current cluster data
    }
  }

  // Output to terminal and log file
  std::cout << output.str();
  logfile_ << output.str();
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(Phase2TrackerDumpDigi);
