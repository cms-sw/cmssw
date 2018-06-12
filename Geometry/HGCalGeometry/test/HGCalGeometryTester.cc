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
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
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
  void doTestWafer(const HGCalGeometry* geom, DetId::Detector det);
  void doTestScint(const HGCalGeometry* geom, DetId::Detector det);
  
  std::string    name;
};

HGCalGeometryTester::HGCalGeometryTester(const edm::ParameterSet& iC) {
  name       = iC.getParameter<std::string>("Detector");
}


HGCalGeometryTester::~HGCalGeometryTester() {}

void HGCalGeometryTester::analyze(const edm::Event& , 
				  const edm::EventSetup& iSetup ) {

  edm::ESHandle<HGCalGeometry> geomH;
  iSetup.get<IdealGeometryRecord>().get(name,geomH);
  const HGCalGeometry* geom = (geomH.product());
  if (!geomH.isValid()) {
    std::cout << "Cannot get valid HGCalGeometry Object for " << name 
	      << std::endl;
  } else {
    HGCalGeometryMode::GeometryMode mode = geom->topology().dddConstants().geomMode();
    if ((mode == HGCalGeometryMode::Hexagon) ||
	(mode == HGCalGeometryMode::HexagonFull)) {
      ForwardSubdetector subdet;
      if      (name == "HGCalHESiliconSensitive")      subdet = HGCHEF;
      else if (name == "HGCalHEScintillatorSensitive") subdet = HGCHEB;
      else                                             subdet = HGCEE;
      std::cout << "Perform test for " << name << " Detector:Subdetector "
		<< DetId::Forward << ":" << subdet << std::endl;
      doTest(geom, subdet);
    } else {
      DetId::Detector det;
      if      (name == "HGCalHESiliconSensitive")      det = DetId::HGCalHSi;
      else if (name == "HGCalHEScintillatorSensitive") det = DetId::HGCalHSc;
      else                                             det = DetId::HGCalEE;
      std::cout << "Perform test for " << name << " Detector " << det
		<< std::endl;
      if (name == "HGCalHEScintillatorSensitive") {
	doTestScint(geom, det);
      } else {
	doTestWafer(geom,det);
      }
    }
  }
}

void HGCalGeometryTester::doTest(const HGCalGeometry* geom, 
				 ForwardSubdetector subdet) {
  
  const std::vector<DetId>& ids = geom->getValidDetIds();
  std::cout << "doTest: " << ids.size() << " valid ids for " 
	    << geom->cellElement() << std::endl;

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
 		      << " parameter[3] = " << icell1->param()[2] << ":"
		      << icell1->param()[2];
	    if (id1.rawId() != idc1.rawId()) 
	      std::cout << "***** ERROR *****" << std::endl;
	    else
	      std::cout << std::endl;
	    std::vector<GlobalPoint> corners = geom->getCorners(idc1);
	    std::cout << corners.size() << " corners";
	    for (auto const & cor : corners)
	      std::cout << " [" << cor.x() << "," << cor.y() << "," << cor.z()
			<< "]";
	    std::cout << std::endl;
	  }
	}
      }
    }
  }
}

void HGCalGeometryTester::doTestWafer(const HGCalGeometry* geom, 
				      DetId::Detector det) {
  
  const std::vector<DetId>& ids = geom->getValidDetIds();
  std::cout << "doTestWafer:: " << ids.size() << " valid ids for " 
	    << geom->cellElement() << std::endl;
  int layers[] = {1, 5, 10};
  int zsides[] = {1, -1};
  int cells[]  = {1, 4, 7};
  int wafers[] = {7, 5, 3,-3, -5, -7};
  for (int zside : zsides) {
    for (int layer : layers) {
      for (int waferU : wafers) {
	for (int waferV : wafers) {
	  int type = geom->topology().dddConstants().getTypeHex(layer,waferU,waferV);
	  std::cout << "zside " << zside << " layer " << layer << " wafer " << waferU << ":" << waferV << " type " << type << std::endl;
	  for (int cellU : cells) {
	    for (int cellV : cells) {
	      std::cout << "det " << det << " cell " << cellU << ":" << cellV << std::endl;
	      DetId id1 = (DetId)(HGCSiliconDetId(det,zside,type,layer,waferU,waferV,cellU,cellV));
	      std::cout << HGCSiliconDetId(id1) << std::endl;
	      if (geom->topology().valid(id1)) {
		auto        icell1  = geom->getGeometry(id1);
		GlobalPoint global1 = geom->getPosition(id1);
		DetId       idc1    = geom->getClosestCell(global1);
		std::cout << "DetId (" << det << ":" << zside << ":" 
			  << type << ":" << layer << ":" << waferU << ":"
			  << waferV << ":" << cellU << ":" << cellV 
			  << ") Geom " << icell1 << " position (" 
			  << global1.x() << ", " << global1.y()
			  << ", " << global1.z() << ") ids " << std::hex 
			  << id1.rawId() << ":" << idc1.rawId() << std::dec
			  << ":" << HGCSiliconDetId(id1) << ":" 
			  << HGCSiliconDetId(idc1) << " parameter[3] = " 
			  << icell1->param()[2] << ":" << icell1->param()[2];
		if (id1.rawId() != idc1.rawId()) 
		  std::cout << "***** ERROR *****" << std::endl;
		else
		  std::cout << std::endl;
		std::vector<GlobalPoint> corners = geom->getCorners(idc1);
		std::cout << corners.size() << " corners";
		for (auto const & cor : corners)
		  std::cout << " [" << cor.x() << "," << cor.y() << "," 
			    << cor.z() << "]";
		std::cout << std::endl;
	      }
	    }
	  }
	}
      }
    }
  }
}
 
void HGCalGeometryTester::doTestScint(const HGCalGeometry* geom, 
				      DetId::Detector det) {
  
  const std::vector<DetId>& ids = geom->getValidDetIds();
  std::cout << "doTestScint: " << ids.size() << " valid ids for " 
	    << geom->cellElement() << std::endl;
  int layers[] = {9, 15, 22};
  int zsides[] = {1, -1};
  int iphis[]  = {1, 51, 101, 151, 201};
  int ietas[]  = {11, 20, 29};
  for (int zside : zsides) {
    for (int layer : layers) {
      int type = geom->topology().dddConstants().getTypeTrap(layer);
      for (int ieta : ietas) {
	for (int iphi : iphis) {
	  DetId id1 = (DetId)(HGCScintillatorDetId(type,layer,zside*ieta,iphi));
	  if (geom->topology().valid(id1)) {
	    auto        icell1  = geom->getGeometry(id1);
	    GlobalPoint global1 = geom->getPosition(id1);
	    DetId       idc1    = geom->getClosestCell(global1);
	    std::cout << "DetId (" << det << ":" << zside << ":" << type 
		      << ":" << layer << ":" << ieta << ":" << iphi
		      << ") Geom " << icell1 << " position (" << global1.x() 
		      << ", " << global1.y() << ", " << global1.z() 
		      << ") ids " << std::hex  << id1.rawId() << ":" 
		      << idc1.rawId() << std::dec << ":" 
		      << HGCScintillatorDetId(id1) << ":" 
		      << HGCScintillatorDetId(idc1)
		      << " parameter[11] = " << icell1->param()[10] << ":"
		      << icell1->param()[10];
	    if (id1.rawId() != idc1.rawId()) 
	      std::cout << "***** ERROR *****" << std::endl;
	    else
	      std::cout << std::endl;
	    std::vector<GlobalPoint> corners = geom->getCorners(idc1);
	    std::cout << corners.size() << " corners";
	    for (auto const & cor : corners)
	      std::cout << " [" << cor.x() << "," << cor.y() << "," 
			<< cor.z() << "]";
	    std::cout << std::endl;
	  }
	}
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalGeometryTester);
