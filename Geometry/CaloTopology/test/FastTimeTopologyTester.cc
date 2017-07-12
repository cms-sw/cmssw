#include <iostream>
#include <string>
#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalCommonData/interface/FastTimeDDDConstants.h"
#include "Geometry/CaloTopology/interface/FastTimeTopology.h"
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"

class FastTimeTopologyTester : public edm::EDAnalyzer {
public:
  explicit FastTimeTopologyTester(const edm::ParameterSet& );
  ~FastTimeTopologyTester() override;

  
  void analyze(const edm::Event&, const edm::EventSetup& ) override;
  void doTest(const FastTimeTopology& topology);

private:
  // ----------member data ---------------------------
};

FastTimeTopologyTester::FastTimeTopologyTester(const edm::ParameterSet& ) {}


FastTimeTopologyTester::~FastTimeTopologyTester() {}

void FastTimeTopologyTester::analyze(const edm::Event& , 
				 const edm::EventSetup& iSetup ) {

  edm::ESHandle<FastTimeTopology> topo;
  iSetup.get<IdealGeometryRecord>().get("FastTimeBarrel",topo);
  if (topo.isValid()) doTest(*topo);
  else                std::cout << "Cannot get a valid FastTimeTopology Object for FastTimeBarrel\n";
}

void FastTimeTopologyTester::doTest(const FastTimeTopology& topology) {
  
  for (int izz=0; izz<=1; izz++) {
    int iz = (2*izz-1);
    for (int eta=1; eta<=265; ++eta) {
      for (int phi=1; phi<=720; ++phi) {
	const FastTimeDetId id(1,eta,phi,iz);
	if (topology.valid(id)) {
	  std::cout << "Neighbours for Tower "  << id << std::endl;
	  std::vector<DetId> idE = topology.east(id);
	  std::vector<DetId> idW = topology.west(id);
	  std::vector<DetId> idN = topology.north(id);
	  std::vector<DetId> idS = topology.south(id);
	  std::cout << "          " << idE.size() << " sets along East:";
	  for (auto & i : idE) 
	    std::cout << " " << (FastTimeDetId)(i());
	  std::cout << std::endl;
	  std::cout << "          " << idW.size() << " sets along West:";
	  for (auto & i : idW) 
	    std::cout << " " << (FastTimeDetId)(i());
	  std::cout << std::endl;
	  std::cout << "          " << idN.size() << " sets along North:";
	  for (auto & i : idN) 
	    std::cout << " " << (FastTimeDetId)(i());
	  std::cout << std::endl;
	  std::cout << "          " << idS.size() << " sets along South:";
	  for (auto & i : idS) 
	    std::cout << " " << (FastTimeDetId)(i());
	  std::cout << std::endl;
	}
	phi += 10;
      }
      eta += 5;
    }
  }

}

//define this as a plug-in
DEFINE_FWK_MODULE(FastTimeTopologyTester);
