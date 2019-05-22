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
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "CoralBase/Exception.h"

class HGCalSizeTester : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalSizeTester(const edm::ParameterSet& );
  ~HGCalSizeTester() override;

  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  
private:
  void doTestWafer(const HGCalGeometry* geom, DetId::Detector det);
  void doTestScint(const HGCalGeometry* geom, DetId::Detector det);
  
  std::string    name;
  edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> geomToken_;
};

HGCalSizeTester::HGCalSizeTester(const edm::ParameterSet& iC):
  name{iC.getParameter<std::string>("detector")},
  geomToken_{esConsumes<HGCalGeometry, IdealGeometryRecord>(edm::ESInputTag{"", name})}
{}


HGCalSizeTester::~HGCalSizeTester() {}

void HGCalSizeTester::analyze(const edm::Event& , 
			      const edm::EventSetup& iSetup ) {

  const auto& geomR = iSetup.getData(geomToken_);
  const HGCalGeometry* geom = &geomR;
  HGCalGeometryMode::GeometryMode mode = geom->topology().dddConstants().geomMode();
  if ((mode == HGCalGeometryMode::Hexagon) ||
      (mode == HGCalGeometryMode::HexagonFull)) {
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

void HGCalSizeTester::doTestWafer(const HGCalGeometry* geom, 
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
		std::vector<GlobalPoint> corners = geom->getNewCorners(idc1);
		std::cout << corners.size() << " corners";
		for (auto const & cor : corners)
		  std::cout << " [" << cor.x() << "," << cor.y() << "," 
			    << cor.z() << "]";
		std::cout << std::endl;
		std::vector<DetId> ids = geom->topology().neighbors(id1);
		int k(0);
		for (auto const& id : ids) {
		  GlobalPoint global0 = geom->getPosition(id);
		  std::cout << "Neighbor[" << k << "] "
			    << HGCSiliconDetId(id) <<  " position (" 
			    << global0.x() << ", " << global0.y() << ", " 
			    << global0.z() << ")" << std::endl;
		  ++k;
		}
	      }
	    }
	  }
	}
      }
    }
  }
}
 
void HGCalSizeTester::doTestScint(const HGCalGeometry* geom, 
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
	    std::vector<GlobalPoint> corners = geom->getNewCorners(idc1);
	    std::cout << corners.size() << " corners";
	    for (auto const & cor : corners)
	      std::cout << " [" << cor.x() << "," << cor.y() << "," 
			<< cor.z() << "]";
	    std::cout << std::endl;
	    std::vector<DetId> ids = geom->topology().neighbors(id1);
	    int k(0);
	    for (auto const& id : ids) {
	      GlobalPoint global0 = geom->getPosition(id);
	      std::cout << "Neighbor[" << k << "] "
			<< HGCScintillatorDetId(id) <<  " position (" 
			<< global0.x() << ", " << global0.y() << ", " 
			<< global0.z() << ")" << std::endl;
	      ++k;
	    }
	  }
	}
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalSizeTester);
