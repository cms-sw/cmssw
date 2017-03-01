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
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/CaloTopology/interface/HGCalTopology.h"
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"

class HGCalTopologyTester : public edm::EDAnalyzer {
public:
  explicit HGCalTopologyTester(const edm::ParameterSet& );
  ~HGCalTopologyTester();

  
  virtual void analyze(const edm::Event&, const edm::EventSetup& );
  void doTest(const HGCalTopology& topology);

private:
  // ----------member data ---------------------------
};

HGCalTopologyTester::HGCalTopologyTester(const edm::ParameterSet& ) {}


HGCalTopologyTester::~HGCalTopologyTester() {}

void HGCalTopologyTester::analyze(const edm::Event& , 
				 const edm::EventSetup& iSetup ) {

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
	      for (unsigned int i=0; i<idE.size(); ++i) 
		std::cout << " " << (HGCEEDetId)(idE[i]());
	      std::cout << std::endl;
	      std::cout << "          " << idW.size() << " sets along West:";
	      for (unsigned int i=0; i<idW.size(); ++i) 
		std::cout << " " << (HGCEEDetId)(idW[i]());
	      std::cout << std::endl;
	      std::cout << "          " << idN.size() << " sets along North:";
	      for (unsigned int i=0; i<idN.size(); ++i) 
		std::cout << " " << (HGCEEDetId)(idN[i]());
	      std::cout << std::endl;
	      std::cout << "          " << idS.size() << " sets along South:";
	      for (unsigned int i=0; i<idS.size(); ++i) 
		std::cout << " " << (HGCEEDetId)(idS[i]());
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
