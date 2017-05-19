#include <iostream>
#include <string>
#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalGeometry/interface/FastTimeGeometry.h"
#include "DataFormats/ForwardDetId/interface/FastTimeDetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "CoralBase/Exception.h"

class FastTimeGeometryTester : public edm::one::EDAnalyzer<> {
public:
  explicit FastTimeGeometryTester(const edm::ParameterSet& );
  ~FastTimeGeometryTester();

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}
  
private:
  void doTest(const FastTimeGeometry& geom, ForwardSubdetector subdet);
  
  std::string    name_;
  int            type_;
};

FastTimeGeometryTester::FastTimeGeometryTester(const edm::ParameterSet& iC) {
  name_ = iC.getParameter<std::string>("Detector");
  type_ = (name_ == "FastTimeBarrel") ? 1 : 2;
}


FastTimeGeometryTester::~FastTimeGeometryTester() {}

void FastTimeGeometryTester::analyze(const edm::Event& , 
				     const edm::EventSetup& iSetup ) {

  ForwardSubdetector subdet = FastTime;

  edm::ESHandle<FastTimeGeometry> geom;
  iSetup.get<IdealGeometryRecord>().get(name_,geom);

  if (geom.isValid()) doTest(*geom, subdet);
  else                std::cout << "Cannot get valid FastTimeGeometry Object "
				<< "for " << name_ << std::endl;
}

void FastTimeGeometryTester::doTest(const FastTimeGeometry& geom, 
				    ForwardSubdetector subdet) {
  
  const std::vector<DetId>& ids = geom.getValidDetIds();
  std::cout << ids.size() << " valid ids for " << geom.cellElement() 
	    << std::endl;

  int iEtaZ[]  = {1, 7, 13};
  int iPhis[]  = {1, 5, 10};
  int zsides[] = {1, -1};
  for (int iz = 0; iz < 2; ++iz) {
    int zside = zsides[iz];
    for (int ie = 0; ie < 3; ++ie) {
      int etaZ = iEtaZ[ie];
      for (int ip = 0; ip < 3; ++ip) {
	int phi  = iPhis[ip];
	DetId id1 = (DetId)(FastTimeDetId(type_,etaZ,phi,zside));
	const CaloCellGeometry* icell1 = geom.getGeometry(id1);
	GlobalPoint global1 = geom.getPosition(id1);
	DetId       idc1    = geom.getClosestCell(global1);
	std::cout << "Input " << FastTimeDetId(id1) << " geometry " << icell1
		  << " position (" << global1.x() << ", " << global1.y() 
		  << ", " << global1.z() << " Output " << FastTimeDetId(idc1);
	if (id1.rawId() != idc1.rawId()) std::cout << "  ***** ERROR *****";
	std::cout << std::endl;
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(FastTimeGeometryTester);
