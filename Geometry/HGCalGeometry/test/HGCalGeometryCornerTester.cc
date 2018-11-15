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
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"

class HGCalGeometryCornerTester : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalGeometryCornerTester(const edm::ParameterSet& );
  ~HGCalGeometryCornerTester() override;

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}
  
private:
  void doTest(const HGCalGeometry* geom);
  const std::string    name_;
  const int            cornerType_;
};

HGCalGeometryCornerTester::HGCalGeometryCornerTester(const edm::ParameterSet& iC) : 
  name_(iC.getParameter<std::string>("detector")),
  cornerType_(iC.getParameter<int>("cornerType")) { }

HGCalGeometryCornerTester::~HGCalGeometryCornerTester() {}

void HGCalGeometryCornerTester::analyze(const edm::Event& , 
					const edm::EventSetup& iSetup ) {

  edm::ESHandle<HGCalGeometry> geomH;
  iSetup.get<IdealGeometryRecord>().get(name_,geomH);
  const HGCalGeometry* geom = (geomH.product());
  if (!geomH.isValid()) {
    std::cout << "Cannot get valid HGCalGeometry Object for " << name_
	      << std::endl;
  } else {
    doTest(geom);
  }
}

void HGCalGeometryCornerTester::doTest(const HGCalGeometry* geom) {
  
  const std::vector<DetId>& ids = geom->getValidDetIds();
  std::cout << "doTest: " << ids.size() << " valid ids for " 
	    << geom->cellElement() << std::endl;
  unsigned int kount(0);
  for (auto const& id : ids) {
    std::cout << "[" << kount << "] ";
    if      (id.det() == DetId::Forward)  std::cout <<HGCalDetId(id);
    else if (id.det() == DetId::HGCalHSc) std::cout <<HGCScintillatorDetId(id);
    else                                  std::cout <<HGCSiliconDetId(id);
    std::cout << std::endl;
    const auto cor = ((cornerType_ > 0) ? geom->getCorners(id) : 
		      ((cornerType_ < 0) ? geom->get8Corners(id) :
		       geom->getNewCorners(id)));
    GlobalPoint gp = geom->getPosition(id);
    std::cout << "Center" << gp << "with " << cor.size() << " Corners: ";
    for (unsigned k=0; k<cor.size(); ++k)
      std::cout << "[" << k << "]" << cor[k];
    std::cout << std::endl;
    ++kount;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalGeometryCornerTester);
