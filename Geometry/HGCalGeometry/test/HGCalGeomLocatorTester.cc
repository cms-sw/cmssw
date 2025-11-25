#include <iomanip>
#include <fstream>
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
  void doTestSilicon(const HGCalGeometry* geom, DetId::Detector det, const char* file1, const  char* file2);
  void doTestScintillator(const HGCalGeometry* geom, DetId::Detector det);

  std::string name_, tag_;
  const unsigned int stepSi_, stepSc_;
  edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> geomToken_;
};

HGCalGeomLocaterTester::HGCalGeomLocaterTester(const edm::ParameterSet& iC)
    : name_{iC.getParameter<std::string>("detector")},
      tag_{iC.getParameter<std::string>("tag")},
      stepSi_{iC.getParameter<uint32_t>("stepSilicon")},
      stepSc_{iC.getParameter<uint32_t>("stepScintillator")},
      geomToken_{esConsumes<HGCalGeometry, IdealGeometryRecord>(edm::ESInputTag{"", name_})} {
	edm::LogVerbatim("HGCalGeomX") << "Detector " << name_ << " Steps " << stepSi_ << ":" << stepSc_ << " Tag " << tag_;
}

void HGCalGeomLocaterTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("detector", "HGCalEESensitive");
  desc.add<std::string>("tag", "EE");
  desc.add<uint32_t>("stepSilicon", 10);
  desc.add<uint32_t>("stepScintillator", 10);
  descriptions.add("hgcalGeomLocatorTesterEE", desc);
}

void HGCalGeomLocaterTester::analyze(const edm::Event&, const edm::EventSetup& iSetup) {
  const HGCalGeometry* geom = &iSetup.getData(geomToken_);
  std::string file1 = "ErrorAll" + tag_ + ".txt";
  std::string file2 = "ErrorWafer" + tag_ + ".txt";
  if (geom->topology().waferHexagon8()) {
    DetId::Detector det;
    if (name_ == "HGCalHESiliconSensitive")
      det = DetId::HGCalHSi;
    else
      det = DetId::HGCalEE;
    edm::LogVerbatim("HGCalGeomX") << "Perform test for " << name_ << " Detector " << det << " Mode "
                                   << geom->topology().dddConstants().geomMode();
    doTestSilicon(geom, det, file1.c_str(), file2.c_str());
  } else if (geom->topology().tileTrapezoid()) {
    DetId::Detector det(DetId::HGCalHSc);
    edm::LogVerbatim("HGCalGeomX") << "Perform test for " << name_ << " Detector " << det << " Mode "
                                   << geom->topology().dddConstants().geomMode();
    doTestScintillator(geom, det);
  }
}

void HGCalGeomLocaterTester::doTestSilicon(const HGCalGeometry* geom, DetId::Detector det, const char* file1, const char* file2) {
  const std::vector<DetId>& ids = geom->getValidDetIds();
  edm::LogVerbatim("HGCalGeomX") << "doTest:: " << ids.size() << " valid ids for " << geom->cellElement();
  const double tol = 0.001;
  std::ofstream fout1(file1);
  std::ofstream fout2(file2, std::ios::out);
  int all(0), good(0), bad(0), badw(0);
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
      DetId id2 = geom->getClosestCell(global, false);
      if (id.rawId() != id2.rawId()) {
        HGCSiliconDetId idn(id2);
        std::string c1 = (id.layer() == idn.layer()) ? "" : " Layer MisMatch";
        std::string c2 = ((id.waferU() == idn.waferU()) && (id.waferV() == idn.waferV())) ? "" : " Wafer Mismatch";
        std::string c3 = ((id.cellU() == idn.cellU()) && (id.cellV() == idn.cellV())) ? "" : " Cell Mismatch";
        st1 << " ***** ERROR *****"
            << " New " << idn << c1 << c2 << c3;
        ++bad;
        geom->topology().dddConstants().locateCell(id, false, true);
	fout1 << " " << id.det() << " " << id.zside() << " " << id.type() << " " << id.layer() << " " << id.waferU() << " " << id.waferV() << " " << id.cellU() << " " << id.cellV() << std::endl;
	if ((id.waferU() != idn.waferU()) || (id.waferV() != idn.waferV())) {
	  fout2 << " " << id.det() << " " << id.zside() << " " << id.type() << " " << id.layer() << " " << id.waferU() << " " << id.waferV() << " " << id.cellU() << " " << id.cellV() << std::endl;
	  ++badw;
	}
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
  fout1.close(); fout2.close();
  edm::LogVerbatim("HGCalGeomX") << "Enters " << bad << " IDs in " << file1 << " and " << badw << " IDs in " << file2;
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
