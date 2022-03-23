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
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"

class HGCalGeometryCornerTester : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalGeometryCornerTester(const edm::ParameterSet&);
  ~HGCalGeometryCornerTester() override = default;

  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;

private:
  void doTest(const HGCalGeometry* geom);
  const edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> geomToken_;
  const int cornerType_;
};

HGCalGeometryCornerTester::HGCalGeometryCornerTester(const edm::ParameterSet& iC)
    : geomToken_{esConsumes<HGCalGeometry, IdealGeometryRecord>(
          edm::ESInputTag{"", iC.getParameter<std::string>("detector")})},
      cornerType_(iC.getParameter<int>("cornerType")) {}

void HGCalGeometryCornerTester::analyze(const edm::Event&, const edm::EventSetup& iSetup) {
  const HGCalGeometry& geom = iSetup.getData(geomToken_);
  doTest(&geom);
}

void HGCalGeometryCornerTester::doTest(const HGCalGeometry* geom) {
  const std::vector<DetId>& ids = geom->getValidDetIds();
  edm::LogVerbatim("HGCalGeomX") << "doTest: " << ids.size() << " valid ids for " << geom->cellElement();
  unsigned int kount(0);
  for (auto const& id : ids) {
    std::ostringstream st1;
    st1 << "[" << kount << "] ";
    if (id.det() == DetId::Forward)
      st1 << HGCalDetId(id);
    else if (id.det() == DetId::HGCalHSc)
      st1 << HGCScintillatorDetId(id);
    else
      st1 << HGCSiliconDetId(id);
    edm::LogVerbatim("HGCalGeomX") << st1.str();
    const auto cor = ((cornerType_ > 0) ? geom->getCorners(id)
                                        : ((cornerType_ < 0) ? geom->get8Corners(id) : geom->getNewCorners(id)));
    GlobalPoint gp = geom->getPosition(id);
    std::ostringstream st2;
    st2 << "Center" << gp << "with " << cor.size() << " Corners: ";
    for (unsigned k = 0; k < cor.size(); ++k)
      st2 << "[" << k << "]" << cor[k];
    st2 << " Area " << geom->getArea(id);
    edm::LogVerbatim("HGCalGeomX") << st2.str();
    ++kount;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalGeometryCornerTester);
