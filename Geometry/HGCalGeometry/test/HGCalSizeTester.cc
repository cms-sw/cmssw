#include <iostream>
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
#include "Geometry/HGCalCommonData/interface/HGCalGeometryMode.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "CoralBase/Exception.h"

class HGCalSizeTester : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalSizeTester(const edm::ParameterSet&);
  ~HGCalSizeTester() override = default;

  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;

private:
  void doTestWafer(const HGCalGeometry* geom, DetId::Detector det);
  void doTestScint(const HGCalGeometry* geom, DetId::Detector det);

  const std::string name;
  const edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> geomToken_;
};

HGCalSizeTester::HGCalSizeTester(const edm::ParameterSet& iC)
    : name{iC.getParameter<std::string>("detector")},
      geomToken_{esConsumes<HGCalGeometry, IdealGeometryRecord>(edm::ESInputTag{"", name})} {}

void HGCalSizeTester::analyze(const edm::Event&, const edm::EventSetup& iSetup) {
  const auto& geomR = iSetup.getData(geomToken_);
  const HGCalGeometry* geom = &geomR;
  if (!geom->topology().waferHexagon6()) {
    DetId::Detector det;
    if (name == "HGCalHESiliconSensitive")
      det = DetId::HGCalHSi;
    else if (name == "HGCalHEScintillatorSensitive")
      det = DetId::HGCalHSc;
    else
      det = DetId::HGCalEE;
    edm::LogVerbatim("HGCalGeomX") << "Perform test for " << name << " Detector " << det;
    if (name == "HGCalHEScintillatorSensitive") {
      doTestScint(geom, det);
    } else {
      doTestWafer(geom, det);
    }
  }
}

void HGCalSizeTester::doTestWafer(const HGCalGeometry* geom, DetId::Detector det) {
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
              DetId id1 = (DetId)(HGCSiliconDetId(det, zside, type, layer, waferU, waferV, cellU, cellV));
              edm::LogVerbatim("HGCalGeomX") << HGCSiliconDetId(id1);
              if (geom->topology().valid(id1)) {
                auto icell1 = geom->getGeometry(id1);
                GlobalPoint global1 = geom->getPosition(id1);
                DetId idc1 = geom->getClosestCell(global1);
                std::string cherr = (id1.rawId() != idc1.rawId()) ? "***** ERROR *****" : "";
                edm::LogVerbatim("HGCalGeomX")
                    << "DetId (" << det << ":" << zside << ":" << type << ":" << layer << ":" << waferU << ":" << waferV
                    << ":" << cellU << ":" << cellV << ") Geom " << icell1 << " position (" << global1.x() << ", "
                    << global1.y() << ", " << global1.z() << ") ids " << std::hex << id1.rawId() << ":" << idc1.rawId()
                    << std::dec << ":" << HGCSiliconDetId(id1) << ":" << HGCSiliconDetId(idc1)
                    << " parameter[3] = " << icell1->param()[2] << ":" << icell1->param()[2] << cherr;
                std::vector<GlobalPoint> corners = geom->getNewCorners(idc1);
                std::ostringstream st1;
                st1 << corners.size() << " corners";
                for (auto const& cor : corners)
                  st1 << " [" << cor.x() << "," << cor.y() << "," << cor.z() << "]";
                edm::LogVerbatim("HGCalGeomX") << st1.str();
                std::vector<DetId> ids = geom->topology().neighbors(id1);
                int k(0);
                for (auto const& id : ids) {
                  GlobalPoint global0 = geom->getPosition(id);
                  edm::LogVerbatim("HGCalGeomX") << "Neighbor[" << k << "] " << HGCSiliconDetId(id) << " position ("
                                                 << global0.x() << ", " << global0.y() << ", " << global0.z() << ")";
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

void HGCalSizeTester::doTestScint(const HGCalGeometry* geom, DetId::Detector det) {
  const std::vector<DetId>& ids = geom->getValidDetIds();
  edm::LogVerbatim("HGCalGeomX") << "doTestScint: " << ids.size() << " valid ids for " << geom->cellElement();
  int layers[] = {9, 15, 22};
  int zsides[] = {1, -1};
  int iphis[] = {1, 51, 101, 151, 201};
  int ietas[] = {11, 20, 29};
  for (int zside : zsides) {
    for (int layer : layers) {
      int type = geom->topology().dddConstants().getTypeTrap(layer);
      edm::LogVerbatim("HGCalGeomX") << "Scintilltor Layer " << layer << " Side " << zside << " TYPE " << type;
      for (int ieta : ietas) {
        for (int iphi : iphis) {
          DetId id1 = (DetId)(HGCScintillatorDetId(type, layer, zside * ieta, iphi));
          if (geom->topology().valid(id1)) {
            auto icell1 = geom->getGeometry(id1);
            GlobalPoint global1 = geom->getPosition(id1);
            DetId idc1 = geom->getClosestCell(global1);
            std::pair<int, int> typm = geom->topology().dddConstants().tileType(layer, ieta, iphi);
            if (typm.first >= 0) {
              HGCScintillatorDetId detId(id1);
              detId.setType(typm.first);
              detId.setSiPM(typm.second);
              id1 = static_cast<DetId>(detId);
            }
            HGCScintillatorDetId detId(idc1);
            typm = geom->topology().dddConstants().tileType(detId.layer(), detId.ring(), detId.iphi());
            if (typm.first >= 0) {
              detId.setType(typm.first);
              detId.setSiPM(typm.second);
              idc1 = static_cast<DetId>(detId);
            }
            std::string cherr = (id1.rawId() != idc1.rawId()) ? "***** ERROR *****" : "";
            edm::LogVerbatim("HGCalGeomX")
                << "DetId (" << det << ":" << zside << ":" << type << ":" << layer << ":" << ieta << ":" << iphi
                << ") Geom " << icell1 << " position (" << global1.x() << ", " << global1.y() << ", " << global1.z()
                << ") ids " << std::hex << id1.rawId() << ":" << idc1.rawId() << std::dec << ":"
                << HGCScintillatorDetId(id1) << ":" << HGCScintillatorDetId(idc1)
                << " parameter[11] = " << icell1->param()[10] << ":" << icell1->param()[10] << cherr;
            std::vector<GlobalPoint> corners = geom->getNewCorners(idc1);
            std::ostringstream st1;
            st1 << corners.size() << " corners";
            for (auto const& cor : corners)
              st1 << " [" << cor.x() << "," << cor.y() << "," << cor.z() << "]";
            edm::LogVerbatim("HGCalGeomX") << st1.str();
            std::vector<DetId> ids = geom->topology().neighbors(id1);
            int k(0);
            for (auto const& id : ids) {
              GlobalPoint global0 = geom->getPosition(id);
              edm::LogVerbatim("HGCalGeomX") << "Neighbor[" << k << "] " << HGCScintillatorDetId(id) << " position ("
                                             << global0.x() << ", " << global0.y() << ", " << global0.z() << ")";
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
