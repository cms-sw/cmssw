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

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/CaloTopology/interface/HGCalTopology.h"
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"

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
};

HGCalTopologyTester::HGCalTopologyTester(const edm::ParameterSet& ) {}

void HGCalTopologyTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

void HGCalTopologyTester::analyze(edm::Event const&, edm::EventSetup const& iSetup ) {

  edm::ESHandle<HGCalTopology> topo;
  iSetup.get<IdealGeometryRecord>().get("HGCalEESensitive",topo);
  if (topo.isValid()) doTest(*topo);
  else                std::cout << "Cannot get a valid HGCalTopology Object for HGCalEESensitive\n";
}

void HGCalTopologyTester::doTest(const HGCalTopology& topology) {
  
  for (int izz=0; izz<=1; izz++) {
    int iz = (2*izz-1);
    for (int subsec=0; subsec<=1; ++subsec) {
      for (int sec=1; sec<=36; ++sec) {
	for (int lay=1; lay<=10; ++lay) {
	  for (int cell=0; cell<8000; ++cell) {
	    const HGCEEDetId id(HGCEE,iz,lay,sec,subsec,cell);
	    if (topology.valid(id)) {
	      std::cout << "Neighbours for Tower "  << id << std::endl;
	      std::vector<DetId> idE = topology.east(id);
	      std::vector<DetId> idW = topology.west(id);
	      std::vector<DetId> idN = topology.north(id);
	      std::vector<DetId> idS = topology.south(id);
	      std::cout << "          " << idE.size() << " sets along East:";
	      for (auto & i : idE) 
		std::cout << " " << (HGCEEDetId)(i());
	      std::cout << std::endl;
	      std::cout << "          " << idW.size() << " sets along West:";
	      for (auto & i : idW) 
		std::cout << " " << (HGCEEDetId)(i());
	      std::cout << std::endl;
	      std::cout << "          " << idN.size() << " sets along North:";
	      for (auto & i : idN) 
		std::cout << " " << (HGCEEDetId)(i());
	      std::cout << std::endl;
	      std::cout << "          " << idS.size() << " sets along South:";
	      for (auto & i : idS) 
		std::cout << " " << (HGCEEDetId)(i());
	      std::cout << std::endl;
	    }
	    cell += 100;
	  }
	}
      }
    }
  }

}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalTopologyTester);
