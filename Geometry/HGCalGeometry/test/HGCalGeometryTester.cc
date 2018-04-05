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
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "CoralBase/Exception.h"

class HGCalGeometryTester : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalGeometryTester(const edm::ParameterSet& );
  ~HGCalGeometryTester() override;

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}
  
private:
  void doTest(const HGCalGeometry* geom, ForwardSubdetector subdet);
  
  std::string    name;
};

HGCalGeometryTester::HGCalGeometryTester(const edm::ParameterSet& iC) {
  name       = iC.getParameter<std::string>("Detector");
}


HGCalGeometryTester::~HGCalGeometryTester() {}

void HGCalGeometryTester::analyze(const edm::Event& , 
				  const edm::EventSetup& iSetup ) {

  ForwardSubdetector subdet;
  if      (name == "HGCalHESiliconSensitive")      subdet = HGCHEF;
  else if (name == "HGCalHEScintillatorSensitive") subdet = HGCHEB;
  else                                             subdet = HGCEE;

  edm::ESHandle<HGCalGeometry> geomH;
  iSetup.get<IdealGeometryRecord>().get(name,geomH);
  const HGCalGeometry* geom = (geomH.product());

  if (geomH.isValid()) doTest(geom, subdet);
  else                 std::cout << "Cannot get valid HGCalGeometry Object for "
				 << name << std::endl;
}

void HGCalGeometryTester::doTest(const HGCalGeometry* geom, 
				 ForwardSubdetector subdet) {
  
  const std::vector<DetId>& ids = geom->getValidDetIds();
  std::cout << ids.size() << " valid ids for " << geom->cellElement() 
	    << std::endl;

  int layers[] = {1, 5, 10};
  int zsides[] = {1, -1};
  int cells[]  = {1, 51, 101};
  int wafers[] = {1, 101, 201, 301, 401};
  const int ismax(5);
  for (int zside : zsides) {
    for (int is = 0; is < ismax; ++is) {
      int sector = wafers[is];
      int type   = geom->topology().dddConstants().waferTypeT(sector);
      if (type != 1) type = 0;
      for (int layer : layers) {
	for (int cell : cells) {
	  DetId id1;
	  id1 = (DetId)(HGCalDetId(subdet,zside,layer,type,sector,cell));
	  if (geom->topology().valid(id1)) {
	    auto        icell1  = geom->getGeometry(id1);
	    GlobalPoint global1 = geom->getPosition(id1);
	    DetId       idc1    = geom->getClosestCell(global1);
	    std::cout << "DetId (" << subdet << ":" << zside << ":" << layer
		      << ":" << sector << ":0:" << cell << ") Geom " << icell1
		      << " position (" << global1.x() << ", " << global1.y()
		      << ", " << global1.z() << ") ids " << std::hex 
		      << id1.rawId() << ":" << idc1.rawId() << std::dec
		      << ":" << HGCalDetId(id1) << ":" << HGCalDetId(idc1)
		      << " parameter[11] = " << icell1->param()[10] << ":"
		      << icell1->param()[11] << std::endl;
	    if (id1.rawId() != idc1.rawId()) std::cout <<"***** ERROR *****\n";
	  }
	}
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalGeometryTester);
