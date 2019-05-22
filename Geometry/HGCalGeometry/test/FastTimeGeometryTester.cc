#include <iostream>
#include <string>
#include <vector>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalGeometry/interface/FastTimeGeometry.h"
#include "DataFormats/ForwardDetId/interface/FastTimeDetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "CoralBase/Exception.h"

class FastTimeGeometryTester : public edm::one::EDAnalyzer<> {
public:
  explicit FastTimeGeometryTester(const edm::ParameterSet& );
  ~FastTimeGeometryTester() override;

  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  
private:
  void doTest(const FastTimeGeometry* geom, ForwardSubdetector subdet);

  edm::ESGetToken<FastTimeGeometry, IdealGeometryRecord> geomToken_;
  std::string    name_;
  int            type_;
};

FastTimeGeometryTester::FastTimeGeometryTester(const edm::ParameterSet& iC) {
  name_ = iC.getParameter<std::string>("Detector");
  type_ = (name_ == "FastTimeBarrel") ? 1 : 2;
  geomToken_ = esConsumes<FastTimeGeometry, IdealGeometryRecord>(edm::ESInputTag{"", name_});
}


FastTimeGeometryTester::~FastTimeGeometryTester() {}

void FastTimeGeometryTester::analyze(const edm::Event& , 
				     const edm::EventSetup& iSetup ) {

  ForwardSubdetector subdet = FastTime;

  const auto& geomR = iSetup.getData(geomToken_);
  const FastTimeGeometry* geom = &geomR;

  doTest(geom, subdet);
}

void FastTimeGeometryTester::doTest(const FastTimeGeometry* geom, 
				    ForwardSubdetector subdet) {
  
  const std::vector<DetId>& ids = geom->getValidDetIds();
  std::cout << ids.size() << " valid ids for " << geom->cellElement() 
	    << std::endl;

  int iEtaZ[]  = {1, 7, 13};
  int iPhis[]  = {1, 5, 10};
  int zsides[] = {1, -1};
  for (int zside : zsides) {
    for (int etaZ : iEtaZ) {
      for (int phi : iPhis) {
	DetId id1 = (DetId)(FastTimeDetId(type_,etaZ,phi,zside));
	auto icell1 = geom->getGeometry(id1);
	GlobalPoint global1 = geom->getPosition(id1);
	DetId       idc1    = geom->getClosestCell(global1);
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
