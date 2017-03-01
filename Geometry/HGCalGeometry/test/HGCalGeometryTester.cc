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
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "CoralBase/Exception.h"

class HGCalGeometryTester : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalGeometryTester(const edm::ParameterSet& );
  ~HGCalGeometryTester();

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}
  
private:
  void doTest(const HGCalGeometry& geom, ForwardSubdetector subdet);
  
  std::string    name;
  bool           squareCell;
};

HGCalGeometryTester::HGCalGeometryTester(const edm::ParameterSet& iC) {
  name       = iC.getParameter<std::string>("Detector");
  squareCell = iC.getParameter<bool>("SquareCell");
}


HGCalGeometryTester::~HGCalGeometryTester() {}

void HGCalGeometryTester::analyze(const edm::Event& , 
				  const edm::EventSetup& iSetup ) {

  ForwardSubdetector subdet;
  if      (name == "HGCalHESiliconSensitive")      subdet = HGCHEF;
  else if (name == "HGCalHEScintillatorSensitive") subdet = HGCHEB;
  else                                             subdet = HGCEE;

  edm::ESHandle<HGCalGeometry> geom;
  iSetup.get<IdealGeometryRecord>().get(name,geom);

  if (geom.isValid()) doTest(*geom, subdet);
  else                std::cout << "Cannot get valid HGCalGeometry Object for "
				<< name << std::endl;
}

void HGCalGeometryTester::doTest(const HGCalGeometry& geom, 
				 ForwardSubdetector subdet) {
  
  const std::vector<DetId>& ids = geom.getValidDetIds();
  std::cout << ids.size() << " valid ids for " << geom.cellElement() 
	    << std::endl;

  int sectors[]= {1, 7, 13};
  int layers[] = {1, 5, 10};
  int zsides[] = {1, -1};
  int cells[]  = {1, 51, 101};
  int wafers[] = {1, 101, 201, 301, 401};
  int ismax    = (squareCell) ? 3 : 5;
  for (int iz = 0; iz < 2; ++iz) {
    int zside = zsides[iz];
    for (int is = 0; is < ismax; ++is) {
      int sector = (squareCell) ? sectors[is] : wafers[is];
      int type   = (squareCell) ? 0 : geom.topology().dddConstants().waferTypeT(sector);
      if (type != 1) type = 0;
      for (int il = 0; il < 3; ++il) {
	int layer = layers[il];
	for (int ic = 0; ic < 3; ++ic) {
	  int cell = cells[ic];
	  DetId id1;
	  if (squareCell) {
	    id1 = ((subdet == HGCEE) ? 
		   (DetId)(HGCEEDetId(subdet,zside,layer,sector,type,cell)) :
		   (DetId)(HGCHEDetId(subdet,zside,layer,sector,type,cell)));
	  } else {
	    id1 = (DetId)(HGCalDetId(subdet,zside,layer,type,sector,cell));
	  }
	  if (geom.topology().valid(id1)) {
	    const CaloCellGeometry* icell1 = geom.getGeometry(id1);
	    GlobalPoint global1 = geom.getPosition(id1);
	    DetId       idc1    = geom.getClosestCell(global1);
	    std::cout << "DetId (" << subdet << ":" << zside << ":" << layer
		      << ":" << sector << ":0:" << cell << ") Geom " << icell1
		      << " position (" << global1.x() << ", " << global1.y()
		      << ", " << global1.z() << ") ids " << std::hex 
		      << id1.rawId() << ":" << idc1.rawId() << std::dec;
	    if (squareCell) {
	      if (subdet == HGCEE)
		std::cout << ":" << HGCEEDetId(id1) << ":" << HGCEEDetId(idc1);
	      else
		std::cout << ":" << HGCHEDetId(id1) << ":" << HGCHEDetId(idc1);
	    } else {
	      std::cout << ":" << HGCalDetId(id1) << ":" << HGCalDetId(idc1);
	    }
	    std::cout << " parameter[11] = " << icell1->param()[10] << ":"
		      << icell1->param()[11] << std::endl;
	    if (id1.rawId() != idc1.rawId()) std::cout <<"***** ERROR *****\n";
	    if (squareCell) {
	      DetId id2= ((subdet == HGCEE) ? 
			  (DetId)(HGCEEDetId(subdet,zside,layer,sector,1,cell)) :
			  (DetId)(HGCHEDetId(subdet,zside,layer,sector,1,cell)));
	      
	      const CaloCellGeometry* icell2 = geom.getGeometry(id2);
	      GlobalPoint global2 = geom.getPosition(id2);
	      DetId       idc2    = geom.getClosestCell(global2);
	      std::cout << "DetId (" << subdet << ":" << zside << ":" << layer
			<< ":" << sector << ":1:" << cell << ") Geom " << icell2
			<< " position (" << global2.x() << ", " << global2.y()
			<< ", " << global2.z() << ") ids " << std::hex 
			<< id2.rawId() << ":" << idc2.rawId() << std::dec 
			<< " parameter[11] = " << icell2->param()[10] << ":"
			<< icell2->param()[11] << std::endl;
	      if (id2.rawId() != idc2.rawId()) std::cout << "***** ERROR *****\n";
	    }
	  }
	}
      }
    }
  }
  if (squareCell) {
    uint32_t probids[] = {1711603886, 1711603890, 1761408735, 1761411303,
			  1801744385, 1805447194};
    for (int k=0; k<6; ++k) {
      DetId id(probids[k]);
      if (id.det() == DetId::Forward && id.subdetId() == (int)(subdet)) {
	if (subdet == HGCEE) std::cout << "Test " << HGCEEDetId(id) << std::endl;
	else                 std::cout << "Test " << HGCHEDetId(id) << std::endl;
	const CaloCellGeometry* icell  = geom.getGeometry(id);
	GlobalPoint             global = geom.getPosition(id);
	std::cout << "Geom Cell: " << icell << " position (" << global.x() 
		  << ", " << global.y() << ", " << global.z() << ")"<< std::endl;
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalGeometryTester);
