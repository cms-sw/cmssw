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
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "CoralBase/Exception.h"

class HGCalGeomLocaterTester : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalGeomLocaterTester(const edm::ParameterSet&);

  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;

private:
  void doTest(const HGCalGeometry* geom, DetId::Detector det);

  std::string name_;
  edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> geomToken_;
};

HGCalGeomLocaterTester::HGCalGeomLocaterTester(const edm::ParameterSet& iC)
    : name_{iC.getParameter<std::string>("Detector")},
      geomToken_{esConsumes<HGCalGeometry, IdealGeometryRecord>(edm::ESInputTag{"", name_})} {}

void HGCalGeomLocaterTester::analyze(const edm::Event&, const edm::EventSetup& iSetup) {
  const HGCalGeometry* geom = &iSetup.getData(geomToken_);
  if (geom->topology().waferHexagon8()) {
    DetId::Detector det;
    if (name_ == "HGCalHESiliconSensitive")
      det = DetId::HGCalHSi;
    else
      det = DetId::HGCalEE;
    std::cout << "Perform test for " << name_ << " Detector " << det << " Mode "
              << geom->topology().dddConstants().geomMode() << std::endl;
    doTest(geom, det);
  }
}

void HGCalGeomLocaterTester::doTest(const HGCalGeometry* geom, DetId::Detector det) {
  const std::vector<DetId>& ids = geom->getValidDetIds();
  std::cout << "doTestWafer:: " << ids.size() << " valid ids for " << geom->cellElement() << std::endl;
  const double tol = 0.001;
  const unsigned int step = 10;
  int all(0), good(0), bad(0);
  for (unsigned int k = 0; k < ids.size(); k += step) {
    ++all;
    HGCSiliconDetId id(ids[k]);
    std::cout << "ID[" << k << "] " << id;
    GlobalPoint global = geom->getPosition(id);
    auto waferxy = geom->topology().dddConstants().locateCell(id, false);
    double dx = global.x() - waferxy.first;
    double dy = global.y() - waferxy.second;
    std::cout << " position (" << global.x() << ", " << global.y() << ", " << global.z() << ") waferXY ("
              << waferxy.first << ", " << waferxy.second << ") Delta (" << dx << ", " << dy << ")";
    if ((std::abs(dx) > tol) || (std::abs(dy) > tol)) {
      std::cout << "***** ERROR *****" << std::endl;
      ++bad;
      geom->topology().dddConstants().locateCell(id, true);
    } else {
      std::cout << std::endl;
      ++good;
    }
  }
  std::cout << "\n\nStudied " << all << " (" << ids.size() << ") IDs of which " << good << " are good and " << bad
            << " are bad\n\n\n\n";
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalGeomLocaterTester);
