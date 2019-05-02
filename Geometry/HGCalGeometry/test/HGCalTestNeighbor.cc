#include <iostream>
#include <string>
#include <vector>

#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "CoralBase/Exception.h"

class HGCalTestNeighbor : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalTestNeighbor(const edm::ParameterSet& );
  ~HGCalTestNeighbor() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  
private:
  void doTest(const HGCalGeometry* geom, ForwardSubdetector subdet, 
	      const MagneticField *bField);
  void doTestWafer(const HGCalGeometry* geom, DetId::Detector det,
		   const MagneticField *bField);
  void doTestScint(const HGCalGeometry* geom, DetId::Detector det,
		   const MagneticField *bField);
  
  std::string         name_;
  std::vector<double> px_, py_, pz_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> fieldToken_;
  edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> geomToken_;
};

HGCalTestNeighbor::HGCalTestNeighbor(const edm::ParameterSet& iC) {
  name_      = iC.getParameter<std::string>("detector");
  px_        = iC.getParameter<std::vector<double> >("pX");
  py_        = iC.getParameter<std::vector<double> >("pY");
  pz_        = iC.getParameter<std::vector<double> >("pZ");
  fieldToken_ = esConsumes<MagneticField, IdealMagneticFieldRecord>(edm::ESInputTag{});
  geomToken_  = esConsumes<HGCalGeometry, IdealGeometryRecord>(edm::ESInputTag{"", name_});
}

HGCalTestNeighbor::~HGCalTestNeighbor() {}

void HGCalTestNeighbor::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {

  edm::ParameterSetDescription desc;
  desc.add<std::string>("detector", "HGCalEESensitive");
  std::vector<double> pxy = {2.0, 5.0, 10.0, 100.0};
  std::vector<double> pz  = {10.0, 5.0, 20.0, 50.0};
  desc.add<std::vector<double> >("pX",pxy);
  desc.add<std::vector<double> >("pY",pxy);
  desc.add<std::vector<double> >("pZ",pz);
  descriptions.add("hgcalEETestNeighbor",desc);
}


void HGCalTestNeighbor::analyze(const edm::Event& , 
				const edm::EventSetup& iSetup ) {

  const auto& bFieldR = iSetup.getData(fieldToken_);
  const MagneticField *bField = &bFieldR;

  const auto& geomR = iSetup.getData(geomToken_);
  const HGCalGeometry* geom = &geomR;
  HGCalGeometryMode::GeometryMode mode = geom->topology().dddConstants().geomMode();
  if ((mode == HGCalGeometryMode::Hexagon) ||
      (mode == HGCalGeometryMode::HexagonFull)) {
    ForwardSubdetector subdet;
    if      (name_ == "HGCalHESiliconSensitive")      subdet = HGCHEF;
    else if (name_ == "HGCalHEScintillatorSensitive") subdet = HGCHEB;
    else                                             subdet = HGCEE;
    std::cout << "Perform test for " << name_ << " Detector:Subdetector "
              << DetId::Forward << ":" << subdet << std::endl;
    doTest(geom, subdet, bField);
  } else {
    DetId::Detector det;
    if      (name_ == "HGCalHESiliconSensitive")      det = DetId::HGCalHSi;
    else if (name_ == "HGCalHEScintillatorSensitive") det = DetId::HGCalHSc;
    else                                              det = DetId::HGCalEE;
    std::cout << "Perform test for " << name_ << " Detector " << det
              << std::endl;
    if (name_ == "HGCalHEScintillatorSensitive") {
      doTestScint(geom, det, bField);
    } else {
      doTestWafer(geom, det, bField);
    }
  }
}

void HGCalTestNeighbor::doTest(const HGCalGeometry* geom, 
			       ForwardSubdetector subdet,
			       const MagneticField *bField) {
  
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
	    for (unsigned int k=0; k<px_.size(); ++k) {
	      GlobalVector p(px_[k],py_[k],zside*pz_[k]);
	      DetId id2 = geom->neighborZ(id1,p);
	      DetId id3 = geom->neighborZ(id1,bField,1,p);
	      std::cout << "DetId" << HGCalDetId(id1) << " :" << global1
			<< " p" << p << " ID2" << HGCalDetId(id2) << " :" 
			<< geom->getPosition(id2) << " ID3" << HGCalDetId(id3)
			<< " :" << geom->getPosition(id3) << std::endl;
	    }
	  }
	}
      }
    }
  }
}

void HGCalTestNeighbor::doTestWafer(const HGCalGeometry* geom, 
				    DetId::Detector det,
				    const MagneticField *bField) {
  
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
	  for (int cellU : cells) {
	    for (int cellV : cells) {
	      DetId id1 = (DetId)(HGCSiliconDetId(det,zside,type,layer,waferU,waferV,cellU,cellV));
	      if (geom->topology().valid(id1)) {
		auto        icell1  = geom->getGeometry(id1);
		GlobalPoint global1 = geom->getPosition(id1);
		for (unsigned int k=0; k<px_.size(); ++k) {
		  GlobalVector p(px_[k],py_[k],zside*pz_[k]);
		  DetId id2 = geom->neighborZ(id1,p);
		  DetId id3 = geom->neighborZ(id1,bField,1,p);
		  std::cout << "DetId" << HGCSiliconDetId(id1) << " :" 
			    << global1 << " p" << p << " ID2" 
			    << HGCSiliconDetId(id2) << " :" 
			    << geom->getPosition(id2) << " ID3" 
			    << HGCSiliconDetId(id3) << " :" 
			    << geom->getPosition(id3) << std::endl;
		}
	      }
	    }
	  }
	}
      }
    }
  }
}
 
void HGCalTestNeighbor::doTestScint(const HGCalGeometry* geom, 
				    DetId::Detector det,
				    const MagneticField *bField) {
  
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
	    for (unsigned int k=0; k<px_.size(); ++k) {
	      GlobalVector p(px_[k],py_[k],zside*pz_[k]);
	      DetId id2 = geom->neighborZ(id1,p);
	      DetId id3 = geom->neighborZ(id1,bField,1,p);
	      std::cout << "DetId" << HGCScintillatorDetId(id1) << " :" 
			<< global1 << " p" << p << " ID2" 
			<< HGCScintillatorDetId(id2) << " :" 
			<< geom->getPosition(id2) << " ID3" 
			<< HGCScintillatorDetId(id3) << " :" 
			<< geom->getPosition(id3) << std::endl;
	    }
	  }
	}
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalTestNeighbor);
