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

#include "TFile.h"
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
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topoToken_;
  const edm::EDGetTokenT<Phase2TrackerCluster1DCollectionNew> token_;
  const TrackerTopology* tTopo_ = nullptr;
  const TrackerGeometry* tGeom_ = nullptr;
  std::map<int, std::pair<int, int>> stackMap_;

  TFile* file_;
  TTree* tree_;
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
};

Phase2TrackerDumpDigi::Phase2TrackerDumpDigi(const edm::ParameterSet& pset)
    : geomToken_(esConsumes<TrackerGeometry, TrackerDigiGeometryRecord, edm::Transition::BeginRun>()),
      topoToken_(esConsumes<TrackerTopology, TrackerTopologyRcd, edm::Transition::BeginRun>()),
      token_(consumes<Phase2TrackerCluster1DCollectionNew>(pset.getParameter<edm::InputTag>("ProductLabel"))) {

    // Initialize TFile and TTree
    file_ = new TFile("Phase2TrackerDumpDigi_redigi.root", "RECREATE");
    tree_ = new TTree("ClusterTree", "Cluster data from Phase2 Tracker");
    
    tree_->Branch("detId", &detId_, "detId/i");
    tree_->Branch("clusterR", &clusterR_, "clusterR/F");
    tree_->Branch("clusterZ", &clusterZ_, "clusterZ/F");
    tree_->Branch("clusterCenter", &clusterCenter_, "clusterCenter/F");
    tree_->Branch("clusterSize", &clusterSize_, "clusterSize/I");
    tree_->Branch("clusterLocalX", &clusterLocalX_, "clusterLocalX/F");
    tree_->Branch("clusterLocalY", &clusterLocalY_, "clusterLocalY/F");
    tree_->Branch("clusterGlobalX", &clusterGlobalX_, "clusterGlobalX/F");
    tree_->Branch("clusterGlobalY", &clusterGlobalY_, "clusterGlobalY/F");
    tree_->Branch("clusterGlobalZ", &clusterGlobalZ_, "clusterGlobalZ/F");

    // Initialize the log file
    logfile_.open("Phase2TrackerDumpDigi_output.txt");
    if (!logfile_.is_open()) {
        throw cms::Exception("OutputFileError") << "Failed to open log file for writing.";
    }
}

Phase2TrackerDumpDigi::~Phase2TrackerDumpDigi() {
    // Write the TTree to the file and close the file
    file_->cd();
    tree_->Write();
    file_->Close();
    delete file_;

    // Close the log file
    logfile_.close();
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
    output << "detId: " << detId.rawId() << std::endl;

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

      tree_->Fill();  // Fill the tree with current cluster data
    }
  }

  // Output to terminal and log file
  std::cout << output.str();
  logfile_ << output.str();
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(Phase2TrackerDumpDigi);
