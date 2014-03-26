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
#include "Geometry/Records/interface/ShashlikNumberingRecord.h"
#include "Geometry/HGCalCommonData/interface/ShashlikDDDConstants.h"
#include "Geometry/CaloTopology/interface/ShashlikTopology.h"
#include "DataFormats/EcalDetId/interface/EKDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "CoralBase/Exception.h"

class ShashlikTopologyTester : public edm::EDAnalyzer {
public:
  explicit ShashlikTopologyTester(const edm::ParameterSet& );
  ~ShashlikTopologyTester();

  
  virtual void analyze(const edm::Event&, const edm::EventSetup& );
  void doTest(const ShashlikTopology& topology);

private:
  // ----------member data ---------------------------
};

ShashlikTopologyTester::ShashlikTopologyTester(const edm::ParameterSet& ) {}


ShashlikTopologyTester::~ShashlikTopologyTester() {}

void ShashlikTopologyTester::analyze(const edm::Event& , 
				 const edm::EventSetup& iSetup ) {


  edm::ESTransientHandle<DDCompactView> pDD;
  iSetup.get<IdealGeometryRecord>().get( pDD );
  std::cout << "Gets Compact View\n";
  edm::ESHandle<ShashlikDDDConstants> pHSNDC;
  iSetup.get<ShashlikNumberingRecord>().get( pHSNDC );
  std::cout << "Gets Numbering Record\n";
  edm::ESHandle<ShashlikTopology> topo;
  iSetup.get<ShashlikNumberingRecord>().get(topo);
  if (topo.isValid()) doTest(*topo);
  else                std::cout << "Cannot get a valid ShashlikTopology Object\n";
}

void ShashlikTopologyTester::doTest(const ShashlikTopology& topology) {
  
  for (int izz=0; izz<=1; izz++) {
    int ro(0), fib(0);
    int iz = (2*izz-1);
    for (int ix=1; ix<=256; ++ix) {
      for (int iy=1; iy<=256; ++iy) {
	const EKDetId id(ix,iy,fib,ro,iz);
	if (topology.valid(id)) {
	  std::cout << "Neighbours for : (" << ix << "," << iy << ") Tower " 
		    << id << std::endl;
	  std::vector<DetId> idE = topology.east(id);
	  std::vector<DetId> idW = topology.west(id);
	  std::vector<DetId> idN = topology.north(id);
	  std::vector<DetId> idS = topology.south(id);
	  std::cout << "          " << idE.size() << " sets along East:";
	  for (unsigned int i=0; i<idE.size(); ++i) 
	    std::cout << " " << (EKDetId)(idE[i]());
	  std::cout << std::endl;
	  std::cout << "          " << idW.size() << " sets along West:";
	  for (unsigned int i=0; i<idW.size(); ++i) 
	    std::cout << " " << (EKDetId)(idW[i]());
	  std::cout << std::endl;
	  std::cout << "          " << idN.size() << " sets along North:";
	  for (unsigned int i=0; i<idN.size(); ++i) 
	    std::cout << " " << (EKDetId)(idN[i]());
	  std::cout << std::endl;
	  std::cout << "          " << idS.size() << " sets along South:";
	  for (unsigned int i=0; i<idS.size(); ++i) 
	    std::cout << " " << (EKDetId)(idS[i]());
	  std::cout << std::endl;
	}
      }
    }
  }

}

//define this as a plug-in
DEFINE_FWK_MODULE(ShashlikTopologyTester);
