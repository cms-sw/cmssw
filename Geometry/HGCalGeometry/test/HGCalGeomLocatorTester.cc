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

class HGCalGeomLocaterTester : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalGeomLocaterTester(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;

private:
  void doTestSilicon(const HGCalGeometry* geom, DetId::Detector det);
  void doTestScintillator(const HGCalGeometry* geom, DetId::Detector det);

  std::string name_;
  const unsigned int stepSi_, stepSc_;
  edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> geomToken_;
};

HGCalGeomLocaterTester::HGCalGeomLocaterTester(const edm::ParameterSet& iC)
    : name_{iC.getParameter<std::string>("detector")},
      stepSi_{iC.getParameter<uint32_t>("stepSilicon")},
      stepSc_{iC.getParameter<uint32_t>("stepScintillator")},
      geomToken_{esConsumes<HGCalGeometry, IdealGeometryRecord>(edm::ESInputTag{"", name_})} {
  edm::LogVerbatim("HGCalGeomX") << "Detector " << name_ << " Steps " << stepSi_ << ":" << stepSc_;
}

void HGCalGeomLocaterTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("detector", "HGCalEESensitive");
  desc.add<uint32_t>("stepSilicon", 10);
  desc.add<uint32_t>("stepScintillator", 10);
  descriptions.add("hgcalGeomLocatorTesterEE", desc);
}

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
  int all(0), good(0), bad(0);
  for (unsigned int k = 0; k < ids.size(); k += stepSi_) {
    ++all;
    HGCSiliconDetId id(ids[k]);
    std::ostringstream st1;
    st1 << "ID[" << k << "] " << id;
    GlobalPoint global = geom->getPosition(id, false, false);
    auto waferxy = geom->topology().dddConstants().locateCell(id, false, false);
    double dx = global.x() - waferxy.first;
    double dy = global.y() - waferxy.second;
    st1 << " position (" << global.x() << ", " << global.y() << ", " << global.z() << ") waferXY (" << waferxy.first
        << ", " << waferxy.second << ") Delta (" << dx << ", " << dy << ")";
    if ((std::abs(dx) > tol) || (std::abs(dy) > tol)) {
      DetId id2 = geom->getClosestCell(global);
      if (id.rawId() != id2.rawId()) {
        HGCSiliconDetId idn(id2);
        std::string c1 = (id.layer() == idn.layer()) ? "" : " Layer MisMatch";
        std::string c2 = ((id.waferU() == idn.waferU()) && (id.waferV() == idn.waferV())) ? "" : " Wafer Mismatch";
        std::string c3 = ((id.cellU() == idn.cellU()) && (id.cellV() == idn.cellV())) ? "" : " Cell Mismatch";
        st1 << " ***** ERROR *****"
            << " New " << idn << c1 << c2 << c3;
        ++bad;
        geom->topology().dddConstants().locateCell(id, false, true);
      } else {
        ++good;
      }
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
  int all(0), good(0), bad(0);
  for (unsigned int k = 0; k < ids.size(); k += stepSc_) {
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
