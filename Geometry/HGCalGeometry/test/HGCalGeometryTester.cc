#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "CoralBase/Exception.h"

class HGCalGeometryTester : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalGeometryTester(const edm::ParameterSet&);
  ~HGCalGeometryTester() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;

private:
  void doTest(const HGCalGeometry* geom, ForwardSubdetector subdet);
  void doTestWafer(const HGCalGeometry* geom, DetId::Detector det);
  void doTestScint(const HGCalGeometry* geom, DetId::Detector det);

  const std::string name;
  const edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> geomToken_;
};

HGCalGeometryTester::HGCalGeometryTester(const edm::ParameterSet& iC)
    : name{iC.getParameter<std::string>("Detector")},
      geomToken_{esConsumes<HGCalGeometry, IdealGeometryRecord>(edm::ESInputTag{"", name})} {}

void HGCalGeometryTester::analyze(const edm::Event&, const edm::EventSetup& iSetup) {
  const HGCalGeometry* geom = &(iSetup.getData(geomToken_));
  if (geom->topology().waferHexagon6()) {
    ForwardSubdetector subdet;
    if (name == "HGCalHESiliconSensitive")
      subdet = HGCHEF;
    else if (name == "HGCalHEScintillatorSensitive")
      subdet = HGCHEB;
    else
      subdet = HGCEE;
    edm::LogVerbatim("HGCalGeomX") << "Perform test for " << name << " Detector:Subdetector " << DetId::Forward << ":"
                                   << subdet << " Mode " << geom->topology().dddConstants().geomMode();
    doTest(geom, subdet);
  } else {
    DetId::Detector det;
    if (name == "HGCalHESiliconSensitive")
      det = DetId::HGCalHSi;
    else if (name == "HGCalHEScintillatorSensitive")
      det = DetId::HGCalHSc;
    else
      det = DetId::HGCalEE;
    edm::LogVerbatim("HGCalGeomX") << "Perform test for " << name << " Detector " << det << " Mode "
                                   << geom->topology().dddConstants().geomMode();
    if (name == "HGCalHEScintillatorSensitive") {
      doTestScint(geom, det);
    } else {
      doTestWafer(geom, det);
    }
  }
}

void HGCalGeometryTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("Detector", "HGCalEESensitive");
  descriptions.add("hgcalGeometryTesterEE", desc);
}

void HGCalGeometryTester::doTest(const HGCalGeometry* geom, ForwardSubdetector subdet) {
  const std::vector<DetId>& ids = geom->getValidDetIds();
  edm::LogVerbatim("HGCalGeomX") << "doTest: " << ids.size() << " valid ids for " << geom->cellElement();

  int layers[] = {1, 5, 10};
  int zsides[] = {1, -1};
  int cells[] = {1, 51, 101};
  int wafers[] = {1, 101, 201, 301, 401};
  const int ismax(5);
  for (int zside : zsides) {
    for (int is = 0; is < ismax; ++is) {
      int sector = wafers[is];
      int type = geom->topology().dddConstants().waferTypeT(sector);
      if (type != 1)
        type = 0;
      for (int layer : layers) {
        for (int cell : cells) {
          DetId id1;
          id1 = static_cast<DetId>(HGCalDetId(subdet, zside, layer, type, sector, cell));
          if (geom->topology().valid(id1)) {
            auto icell1 = geom->getGeometry(id1);
            GlobalPoint global1 = geom->getPosition(id1);
            DetId idc1 = geom->getClosestCell(global1);
            GlobalPoint global2 = geom->getPosition(idc1);
            std::string cherr = (id1.rawId() != idc1.rawId()) ? " ***** ERROR *****" : "";
            edm::LogVerbatim("HGCalGeomX")
                << "DetId (" << subdet << ":" << zside << ":" << layer << ":" << sector << ":0:" << cell << ") Geom "
                << icell1 << " position (" << global1.x() << ", " << global1.y() << ", " << global1.z() << ") ids "
                << std::hex << id1.rawId() << ":" << idc1.rawId() << std::dec << ":" << HGCalDetId(id1) << ":"
                << HGCalDetId(idc1) << " new position (" << global2.x() << ", " << global2.y() << ", " << global2.z()
                << ") parameter[3] = " << icell1->param()[2] << ":" << icell1->param()[2] << cherr;
            std::vector<GlobalPoint> corners = geom->getCorners(idc1);
            std::ostringstream st1;
            st1 << corners.size() << " corners";
            for (auto const& cor : corners)
              st1 << " [" << cor.x() << "," << cor.y() << "," << cor.z() << "]";
            edm::LogVerbatim("HGCalGeomX") << st1.str();
          }
        }
      }
    }
  }
}

void HGCalGeometryTester::doTestWafer(const HGCalGeometry* geom, DetId::Detector det) {
  const std::vector<DetId>& ids = geom->getValidDetIds();
  edm::LogVerbatim("HGCalGeomX") << "doTestWafer:: " << ids.size() << " valid ids for " << geom->cellElement();
  int layers[] = {1, 5, 10};
  int zsides[] = {1, -1};
  int cells[] = {1, 4, 7};
  int wafers[] = {7, 5, 3, -3, -5, -7};
  for (int zside : zsides) {
    for (int layer : layers) {
      for (int waferU : wafers) {
        for (int waferV : wafers) {
          int type = geom->topology().dddConstants().getTypeHex(layer, waferU, waferV);
          edm::LogVerbatim("HGCalGeomX") << "zside " << zside << " layer " << layer << " wafer " << waferU << ":"
                                         << waferV << " type " << type;
          for (int cellU : cells) {
            for (int cellV : cells) {
              edm::LogVerbatim("HGCalGeomX") << "det " << det << " cell " << cellU << ":" << cellV;
              DetId id1 = static_cast<DetId>(HGCSiliconDetId(det, zside, type, layer, waferU, waferV, cellU, cellV));
              edm::LogVerbatim("HGCalGeomX") << HGCSiliconDetId(id1);
              if (geom->topology().valid(id1)) {
                auto icell1 = geom->getGeometry(id1);
                GlobalPoint global1 = geom->getPosition(id1);
                DetId idc1 = geom->getClosestCell(global1);
                GlobalPoint global2 = geom->getPosition(idc1);
                std::string cherr = (id1.rawId() != idc1.rawId()) ? " ***** ERROR *****" : "";
                edm::LogVerbatim("HGCalGeomX")
                    << "DetId (" << det << ":" << zside << ":" << type << ":" << layer << ":" << waferU << ":" << waferV
                    << ":" << cellU << ":" << cellV << ") Geom " << icell1 << " position (" << global1.x() << ", "
                    << global1.y() << ", " << global1.z() << ") ids " << std::hex << id1.rawId() << ":" << idc1.rawId()
                    << std::dec << ":" << HGCSiliconDetId(id1) << ":" << HGCSiliconDetId(idc1) << " new position ("
                    << global2.x() << ", " << global2.y() << ", " << global2.z()
                    << ") parameter[3] = " << icell1->param()[2] << ":" << icell1->param()[2] << cherr;
                std::vector<GlobalPoint> corners = geom->getCorners(idc1);
                std::ostringstream st1;
                st1 << corners.size() << " corners";
                for (auto const& cor : corners)
                  st1 << " [" << cor.x() << "," << cor.y() << "," << cor.z() << "]";
                edm::LogVerbatim("HGCalGeomX") << st1.str();
              }
            }
          }
        }
      }
    }
  }
}

void HGCalGeometryTester::doTestScint(const HGCalGeometry* geom, DetId::Detector det) {
  const std::vector<DetId>& ids = geom->getValidDetIds();
  edm::LogVerbatim("HGCalGeomX") << "doTestScint: " << ids.size() << " valid ids for " << geom->cellElement();
  int layers[] = {9, 14, 21};
  int zsides[] = {1, -1};
  int iphis[] = {1, 51, 101, 151, 201};
  int ietas[] = {11, 20, 29};
  for (int zside : zsides) {
    for (int layer : layers) {
      int type = geom->topology().dddConstants().getTypeTrap(layer);
      for (int ieta : ietas) {
        std::pair<int, int> typm = geom->topology().dddConstants().tileType(layer, ieta, 0);
        for (int iphi : iphis) {
          HGCScintillatorDetId detId(type, layer, zside * ieta, iphi);
          if (typm.first >= 0) {
            detId.setType(typm.first);
            detId.setSiPM(typm.second);
          }
          DetId id1 = static_cast<DetId>(detId);
          if (geom->topology().valid(id1)) {
            auto icell1 = geom->getGeometry(id1);
            GlobalPoint global1 = geom->getPosition(id1);
            DetId idc1 = geom->getClosestCell(global1);
            GlobalPoint global2 = geom->getPosition(idc1);
            std::string cherr = (id1.rawId() != idc1.rawId()) ? " ***** ERROR *****" : "";
            edm::LogVerbatim("HGCalGeomX")
                << "DetId (" << det << ":" << zside << ":" << type << ":" << layer << ":" << ieta << ":" << iphi
                << ") Geom " << icell1 << " position (" << global1.x() << ", " << global1.y() << ", " << global1.z()
                << ":" << global1.perp() << ") ids " << std::hex << id1.rawId() << ":" << idc1.rawId() << std::dec
                << ":" << HGCScintillatorDetId(id1) << ":" << HGCScintillatorDetId(idc1) << " new position ("
                << global2.x() << ", " << global2.y() << ", " << global2.z() << ":" << global2.perp()
                << ") parameter[11] = " << icell1->param()[10] << ":" << icell1->param()[10] << cherr;
            std::vector<GlobalPoint> corners = geom->getCorners(idc1);
            std::ostringstream st1;
            st1 << corners.size() << " corners";
            for (auto const& cor : corners)
              st1 << " [" << cor.x() << "," << cor.y() << "," << cor.z() << "]";
            edm::LogVerbatim("HGCalGeomX") << st1.str();
          }
        }
      }
    }
  }
  for (int layer = geom->topology().dddConstants().firstLayer();
       layer <= geom->topology().dddConstants().lastLayer(true);
       ++layer) {
    HGCScintillatorDetId id(geom->topology().dddConstants().getTypeTrap(layer), layer, 20, 1);
    GlobalPoint global1 = geom->getPosition(id);
    edm::LogVerbatim("HGCalGeomX") << "Layer " << layer << " DetId " << id << " position (" << global1.x() << ", "
                                   << global1.y() << ", " << global1.z() << ", " << global1.perp() << ")";
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalGeometryTester);
