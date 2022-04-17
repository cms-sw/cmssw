#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "CoralBase/Exception.h"

class HGCalGeomLocaterTester : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalGeomLocaterTester(const edm::ParameterSet&);

  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;

private:
  void doTestSilicon(const HGCalGeometry* geom, DetId::Detector det);
  void doTestScintillator(const HGCalGeometry* geom, DetId::Detector det);

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
    edm::LogVerbatim("HGCalGeomX") << "Perform test for " << name_ << " Detector " << det << " Mode "
                                   << geom->topology().dddConstants().geomMode();
    doTestSilicon(geom, det);
  } else if (geom->topology().tileTrapezoid()) {
    DetId::Detector det(DetId::HGCalHSc);
    edm::LogVerbatim("HGCalGeomX") << "Perform test for " << name_ << " Detector " << det << " Mode "
                                   << geom->topology().dddConstants().geomMode();
    doTestScintillator(geom, det);
  }
}

void HGCalGeomLocaterTester::doTestSilicon(const HGCalGeometry* geom, DetId::Detector det) {
  const std::vector<DetId>& ids = geom->getValidDetIds();
  edm::LogVerbatim("HGCalGeomX") << "doTest:: " << ids.size() << " valid ids for " << geom->cellElement();
  const double tol = 0.001;
  const unsigned int step = 10;
  int all(0), good(0), bad(0);
  for (unsigned int k = 0; k < ids.size(); k += step) {
    ++all;
    HGCSiliconDetId id(ids[k]);
    std::ostringstream st1;
    st1 << "ID[" << k << "] " << id;
    GlobalPoint global = geom->getPosition(id);
    auto waferxy = geom->topology().dddConstants().locateCell(id, false);
    double dx = global.x() - waferxy.first;
    double dy = global.y() - waferxy.second;
    st1 << " position (" << global.x() << ", " << global.y() << ", " << global.z() << ") waferXY (" << waferxy.first
        << ", " << waferxy.second << ") Delta (" << dx << ", " << dy << ")";
    if ((std::abs(dx) > tol) || (std::abs(dy) > tol)) {
      st1 << " ***** ERROR *****";
      ++bad;
      geom->topology().dddConstants().locateCell(id, true);
    } else {
      ++good;
    }
    edm::LogVerbatim("HGCalGeomX") << st1.str();
  }
  edm::LogVerbatim("HGCalGeomX") << "\n\nStudied " << all << " (" << ids.size() << ") IDs of which " << good
                                 << " are good and " << bad << " are bad\n\n\n";
}

void HGCalGeomLocaterTester::doTestScintillator(const HGCalGeometry* geom, DetId::Detector det) {
  const std::vector<DetId>& ids = geom->getValidDetIds();
  edm::LogVerbatim("HGCalGeomX") << "doTest:: " << ids.size() << " valid ids for " << geom->cellElement();
  const double tol = 0.01;
  const unsigned int step = 10;
  int all(0), good(0), bad(0);
  for (unsigned int k = 0; k < ids.size(); k += step) {
    ++all;
    HGCScintillatorDetId id(ids[k]);
    std::ostringstream st1;
    st1 << "ID[" << k << "] " << id;
    GlobalPoint global = geom->getPosition(id);
    auto tilexy = geom->topology().dddConstants().locateCell(id, false);
    double dx = global.x() - tilexy.first;
    double dy = global.y() - tilexy.second;
    st1 << " position (" << global.x() << ", " << global.y() << ", " << global.z() << ") tileXY (" << tilexy.first
        << ", " << tilexy.second << ") Delta (" << dx << ", " << dy << ")";
    if ((std::abs(dx) > tol) || (std::abs(dy) > tol)) {
      double r1 = sqrt(global.x() * global.x() + global.y() * global.y());
      double r2 = sqrt(tilexy.first * tilexy.first + tilexy.second * tilexy.second);
      st1 << " R " << r1 << ":" << r2 << " ***** ERROR *****";
      ++bad;
      geom->topology().dddConstants().locateCell(id, true);
    } else {
      ++good;
    }
    edm::LogVerbatim("HGCalGeomX") << st1.str();
  }
  edm::LogVerbatim("HGCalGeomX") << "\n\nStudied " << all << " (" << ids.size() << ") IDs of which " << good
                                 << " are good and " << bad << " are bad\n\n\n";
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalGeomLocaterTester);
