#include <iostream>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/CaloTopology/interface/HGCalTopology.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"

class HGCalTopologyTester : public edm::one::EDAnalyzer<edm::one::WatchRuns> { 

public:
  explicit HGCalTopologyTester(const edm::ParameterSet& );
  
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void beginJob() override {}
  void beginRun(edm::Run const&, edm::EventSetup const&) override {}
  void endRun(edm::Run const&, edm::EventSetup const&) override {}
  void doTest(const HGCalTopology& topology);

  // ----------member data ---------------------------
  const std::string      detectorName_;
  const std::vector<int> type_, layer_, sec1_, sec2_, cell1_, cell2_;
};

HGCalTopologyTester::HGCalTopologyTester(const edm::ParameterSet& iC) :
  detectorName_(iC.getParameter<std::string>("detectorName")),
  type_(iC.getParameter<std::vector<int> >("types")),
  layer_(iC.getParameter<std::vector<int> >("layers")),
  sec1_(iC.getParameter<std::vector<int> >("sector1")),
  sec2_(iC.getParameter<std::vector<int> >("sector2")),
  cell1_(iC.getParameter<std::vector<int> >("cell1")),
  cell2_(iC.getParameter<std::vector<int> >("cell2")) {}

void HGCalTopologyTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

  edm::ParameterSetDescription desc;
  std::vector<int> types = {0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2};
  std::vector<int> layer = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18};
  std::vector<int> sec1  = {1,1,2,2,3,3,5,5,6,6,7,7,8,8,9,9,10,10};
  std::vector<int> sec2  = {3,3,3,3,2,2,6,6,6,6,3,3,8,8,9,9,3,3};
  std::vector<int> cell1 = {0,4,12,14,18,23,1,4,7,10,13,16,0,3,6,9,12,15};
  std::vector<int> cell2 = {0,4,0,2,23,18,1,4,7,10,13,16,0,3,6,9,12,15};
  desc.add<std::string>("detectorName","HGCalEESensitive");
  desc.add<std::vector<int> >("types",types);
  desc.add<std::vector<int> >("layers",layer);
  desc.add<std::vector<int> >("sector1",sec1);
  desc.add<std::vector<int> >("sector2",sec2);
  desc.add<std::vector<int> >("cell1",cell1);
  desc.add<std::vector<int> >("cell2",cell2);
  descriptions.add("hgcalTopologyTesterEE",desc);
}

void HGCalTopologyTester::analyze(edm::Event const&, edm::EventSetup const& iSetup ) {

  edm::ESHandle<HGCalTopology> topo;
  iSetup.get<IdealGeometryRecord>().get(detectorName_,topo);
  if (topo.isValid()) doTest(*topo);
  else                std::cout << "Cannot get a valid Topology Object for "
				<< detectorName_;
}

void HGCalTopologyTester::doTest(const HGCalTopology& topology) {
  
  if ((topology.geomMode() == HGCalGeometryMode::Hexagon8) ||
      (topology.geomMode() == HGCalGeometryMode::Hexagon8Full) ||
      (topology.geomMode() == HGCalGeometryMode::Trapezoid)) {
    for (unsigned int i=0; i<type_.size(); ++i) {
      DetId id;
      if (detectorName_ == "HGCalEESensitive") {
	id = HGCSiliconDetId(DetId::HGCalEE,1,type_[i],layer_[i],sec1_[i],
			     sec2_[i],cell1_[i],cell2_[i]);
      } else if (detectorName_ == "HGCalHESiliconSensitive") {
	id = HGCSiliconDetId(DetId::HGCalHSi,1,type_[i],layer_[i],sec1_[i],
			     sec2_[i],cell1_[i],cell2_[i]);
      } else if (detectorName_ == "HGCalHEScintillatorSensitive") {
	id = HGCScintillatorDetId(type_[i],layer_[i],sec1_[i],cell1_[i]);
      } else {
	break;
      }
      std::vector<DetId> ids = topology.neighbors(id);
      unsigned int k(0);
      if (id.det() == DetId::HGCalEE || id.det() == DetId::HGCalHSi) {
	std::cout << (HGCSiliconDetId)(id) << " has " << ids.size()
		  << " neighbours:" << std::endl;
	for (const auto & idn : ids) {
	  std::cout << "[" << k << "] " << (HGCSiliconDetId)(idn) << std::endl;
	  ++k;
	}
      } else {
	std::cout << (HGCScintillatorDetId)(id) << " has " << ids.size()
		  << " neighbours:" << std::endl;
	for (const auto & idn : ids) {
	  std::cout << "[" << k << "] " << (HGCScintillatorDetId)(idn) << "\n";
	  ++k;
	}
      }
    }
  }

}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalTopologyTester);
