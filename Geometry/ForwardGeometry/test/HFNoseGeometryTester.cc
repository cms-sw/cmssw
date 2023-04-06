#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HFNoseDetId.h"
#include "CoralBase/Exception.h"

class HFNoseGeometryTester : public edm::one::EDAnalyzer<> {
public:
  explicit HFNoseGeometryTester(const edm::ParameterSet&);
  ~HFNoseGeometryTester() override {}

  void beginJob() override {}
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;
  void endJob() override {}

private:
  void doTestWafer(const HGCalGeometry* geom);

  const std::string name_;
  const edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> tokGeom_;
};

HFNoseGeometryTester::HFNoseGeometryTester(const edm::ParameterSet& iC)
    : name_(iC.getParameter<std::string>("Detector")),
      tokGeom_(esConsumes<HGCalGeometry, IdealGeometryRecord>(edm::ESInputTag{"", name_})) {}

void HFNoseGeometryTester::analyze(const edm::Event&, const edm::EventSetup& iSetup) {
  const HGCalGeometry* geom = &iSetup.getData(tokGeom_);
  if (geom->topology().isHFNose()) {
    doTestWafer(geom);
  } else {
    edm::LogVerbatim("ForwardGeom") << name_ << " is not a valid name for HFNose Detecor";
  }
}

void HFNoseGeometryTester::doTestWafer(const HGCalGeometry* geom) {
  const std::vector<DetId>& ids = geom->getValidDetIds();
  edm::LogVerbatim("ForwardGeom") << "doTestWafer:: " << ids.size() << " valid ids for " << geom->cellElement();
  int layers[] = {1, 4, 7};
  int zsides[] = {1, -1};
  int cells[] = {1, 4, 7};
  int wafers[] = {7, 5, 3, -3, -5, -7};
  for (int zside : zsides) {
    for (int layer : layers) {
      for (int waferU : wafers) {
        for (int waferV : wafers) {
          int type = geom->topology().dddConstants().getTypeHex(layer, waferU, waferV);
          edm::LogVerbatim("ForwardGeom")
              << "zside " << zside << " layer " << layer << " wafer " << waferU << ":" << waferV << " type " << type;
          for (int cellU : cells) {
            for (int cellV : cells) {
              edm::LogVerbatim("ForwardGeom") << " cell " << cellU << ":" << cellV;
              DetId id1 = (DetId)(HFNoseDetId(zside, type, layer, waferU, waferV, cellU, cellV));
              edm::LogVerbatim("ForwardGeom") << HFNoseDetId(id1);
              if (geom->topology().valid(id1)) {
                auto icell1 = geom->getGeometry(id1);
                GlobalPoint global1 = geom->getPosition(id1);
                DetId idc1 = geom->getClosestCell(global1);
                std::ostringstream st1;
                st1 << "DetId (" << zside << ":" << type << ":" << layer << ":" << waferU << ":" << waferV << ":"
                    << cellU << ":" << cellV << ") Geom " << icell1 << " position (" << global1.x() << ", "
                    << global1.y() << ", " << global1.z() << ") ids " << std::hex << id1.rawId() << ":" << idc1.rawId()
                    << std::dec << ":" << HFNoseDetId(id1) << ":" << HFNoseDetId(idc1)
                    << " parameter[3] = " << icell1->param()[2] << ":" << icell1->param()[2];
                if (id1.rawId() != idc1.rawId())
                  st1 << "***** ERROR *****";
                edm::LogVerbatim("ForwardGeom") << st1.str();
                std::vector<GlobalPoint> corners = geom->getCorners(idc1);
                std::ostringstream st2;
                st2 << corners.size() << " corners";
                for (auto const& cor : corners)
                  st2 << " [" << cor.x() << "," << cor.y() << "," << cor.z() << "]";
                edm::LogVerbatim("ForwardGeom") << st2.str();
              }
            }
          }
        }
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HFNoseGeometryTester);
