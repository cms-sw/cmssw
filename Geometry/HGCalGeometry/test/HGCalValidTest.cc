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
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "DataFormats/ForwardDetId/interface/ForwardSubdetector.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"

class HGCalValidTest : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalValidTest(const edm::ParameterSet&);
  ~HGCalValidTest() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;

private:
  const std::string name;
  const edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> geomToken_;
};

HGCalValidTest::HGCalValidTest(const edm::ParameterSet& iC)
    : name{iC.getParameter<std::string>("detector")},
      geomToken_{esConsumes<HGCalGeometry, IdealGeometryRecord>(edm::ESInputTag{"", name})} {}

void HGCalValidTest::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("detector", "HGCalHEScintillatorSensitive");
  descriptions.add("hgcalValidTestHEB", desc);
}

void HGCalValidTest::analyze(const edm::Event&, const edm::EventSetup& iSetup) {
  const auto& geom = &iSetup.getData(geomToken_);
  DetId::Detector det;
  if (geom->topology().waferHexagon6()) {
    ForwardSubdetector subdet;
    if (name == "HGCalHESiliconSensitive")
      subdet = HGCHEF;
    else if (name == "HGCalHEScintillatorSensitive")
      subdet = HGCHEB;
    else
      subdet = HGCEE;
    edm::LogVerbatim("HGCalGeom") << "Cannot perform test for " << name << " Detector:Subdetector " << DetId::Forward
                                  << ":" << subdet << " Mode " << geom->topology().dddConstants().geomMode();
    return;
  } else {
    if (name == "HGCalHESiliconSensitive")
      det = DetId::HGCalHSi;
    else if (name == "HGCalHEScintillatorSensitive")
      det = DetId::HGCalHSc;
    else
      det = DetId::HGCalEE;
    edm::LogVerbatim("HGCalGeom") << "Perform test for " << name << " Detector " << det << " Mode "
                                  << geom->topology().dddConstants().geomMode();
  }

  int firstLayer = geom->topology().dddConstants().firstLayer();
  int lastLayer = geom->topology().dddConstants().lastLayer(true);
  const std::vector<DetId>& ids = geom->getValidDetIds();
  edm::LogVerbatim("HGCalGeom") << "doTest: " << ids.size() << " valid ids for " << geom->cellElement();

  std::vector<int> zsides = {-1, 1};
  for (int zside : zsides) {
    for (int layer = firstLayer; layer <= lastLayer; ++layer) {
      std::vector<std::pair<int, int> > done;
      for (auto const& id : ids) {
        if (det == DetId::HGCalHSc) {
          HGCScintillatorDetId hid(id);
          if ((hid.zside() != zside) || (hid.layer() != layer))
            continue;
          std::pair<int, int> ring = std::make_pair(hid.ring(), 0);
          if (std::find(done.begin(), done.end(), ring) != done.end())
            continue;
          done.emplace_back(ring);
          edm::LogVerbatim("HGCalGeom") << "Corners for " << hid;
        } else {
          HGCSiliconDetId hid(id);
          if ((hid.zside() != zside) || (hid.layer() != layer))
            continue;
          if (std::find(done.begin(), done.end(), hid.waferUV()) != done.end())
            continue;
          done.emplace_back(hid.waferUV());
          edm::LogVerbatim("HGCalGeom") << "Corners for " << hid;
        }

        const auto cor = geom->getNewCorners(id);
        std::ostringstream st1;
        for (unsigned int k = 0; k < cor.size(); ++k)
          st1 << " [" << k << "] " << std::setprecision(4) << cor[k];
        edm::LogVerbatim("HGCalGeom") << st1.str();
      }
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalValidTest);
