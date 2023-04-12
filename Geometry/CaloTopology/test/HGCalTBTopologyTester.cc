#include <iostream>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalTBCommonData/interface/HGCalTBDDDConstants.h"
#include "Geometry/CaloTopology/interface/HGCalTBTopology.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"

class HGCalTBTopologyTester : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit HGCalTBTopologyTester(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void beginJob() override {}
  void beginRun(edm::Run const&, edm::EventSetup const&) override {}
  void endRun(edm::Run const&, edm::EventSetup const&) override {}
  void doTest(const HGCalTBTopology& topology);

  // ----------member data ---------------------------
  const std::string detectorName_;
  const std::vector<int> type_, layer_, sector_, cells_;
  const edm::ESGetToken<HGCalTBTopology, IdealGeometryRecord> tokTopo_;
};

HGCalTBTopologyTester::HGCalTBTopologyTester(const edm::ParameterSet& iC)
    : detectorName_(iC.getParameter<std::string>("detectorName")),
      type_(iC.getParameter<std::vector<int> >("types")),
      layer_(iC.getParameter<std::vector<int> >("layers")),
      sector_(iC.getParameter<std::vector<int> >("sector")),
      cells_(iC.getParameter<std::vector<int> >("cells")),
      tokTopo_{esConsumes<HGCalTBTopology, IdealGeometryRecord>(edm::ESInputTag{"", detectorName_})} {}

void HGCalTBTopologyTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  std::vector<int> types = {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2};
  std::vector<int> layer = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
  std::vector<int> sect = {1, 1, 2, 2, 3, 3, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10};
  std::vector<int> cells = {0, 4, 12, 14, 18, 23, 1, 4, 7, 10, 13, 16, 0, 3, 6, 9, 12, 15};
  desc.add<std::string>("detectorName", "HGCalEESensitive");
  desc.add<std::vector<int> >("types", types);
  desc.add<std::vector<int> >("layers", layer);
  desc.add<std::vector<int> >("sector", sect);
  desc.add<std::vector<int> >("cells", cells);
  descriptions.add("hgcalTBTopologyTesterEE", desc);
}

void HGCalTBTopologyTester::analyze(edm::Event const&, edm::EventSetup const& iSetup) {
  doTest(iSetup.getData(tokTopo_));
}

void HGCalTBTopologyTester::doTest(const HGCalTBTopology& topology) {
  for (unsigned int i = 0; i < type_.size(); ++i) {
    DetId id;
    if (detectorName_ == "HGCalEESensitive") {
      id = HGCalDetId(ForwardSubdetector::HGCEE, 1, layer_[i], type_[i], sector_[i], cells_[i]);
    } else if (detectorName_ == "HGCalHESiliconSensitive") {
      id = HGCalDetId(ForwardSubdetector::HGCHEF, 1, layer_[i], type_[i], sector_[i], cells_[i]);
    } else {
      break;
    }
    if (topology.valid(id)) {
      std::vector<DetId> ids = topology.neighbors(id);
      unsigned int k(0);
      edm::LogVerbatim("HGCalGeom") << static_cast<HGCalDetId>(id) << " has " << ids.size() << " neighbours:";
      for (const auto& idn : ids) {
        edm::LogVerbatim("HGCalGeom") << "[" << k << "] " << (HGCSiliconDetId)(idn);
        ++k;
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalTBTopologyTester);
