#include <iostream>
#include <sstream>
#include <map>
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

class HGCalValidScintTest : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalValidScintTest(const edm::ParameterSet&);
  ~HGCalValidScintTest() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override;

private:
  const edm::ESGetToken<HGCalGeometry, IdealGeometryRecord> geomToken_;
  struct layerInfo {
    int ringMin, ringMax;
    double rMin, rMax;
    layerInfo(int minR = 100, double rMn = 0, int maxR = 0, double rMx = 0)
        : ringMin(minR), ringMax(maxR), rMin(rMn), rMax(rMx){};
  };
};

HGCalValidScintTest::HGCalValidScintTest(const edm::ParameterSet& iC)
    : geomToken_(esConsumes<HGCalGeometry, IdealGeometryRecord>(edm::ESInputTag{"", "HGCalHEScintillatorSensitive"})) {}

void HGCalValidScintTest::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.add("hgcalValidScintTest", desc);
}

void HGCalValidScintTest::analyze(const edm::Event&, const edm::EventSetup& iSetup) {
  const auto& geom = &iSetup.getData(geomToken_);
  DetId::Detector det = DetId::HGCalHSc;
  edm::LogVerbatim("HGCalGeom") << "Perform test for HGCalHEScintillatorSensitive Detector " << det << " Mode "
                                << geom->topology().dddConstants().geomMode();

  int firstLayer = geom->topology().dddConstants().firstLayer();
  int lastLayer = geom->topology().dddConstants().lastLayer(true);
  const std::vector<DetId>& ids = geom->getValidDetIds();
  edm::LogVerbatim("HGCalGeom") << "doTest: " << ids.size() << " valid ids for " << geom->cellElement();

  int zside(1);
  std::map<int, layerInfo> layerMap;
  for (int layer = firstLayer; layer <= lastLayer; ++layer) {
    std::vector<std::pair<int, int> > done;
    for (auto const& id : ids) {
      HGCScintillatorDetId hid(id);
      if ((hid.zside() != zside) || (hid.layer() != layer))
        continue;
      std::pair<int, int> ring = std::make_pair(hid.ring(), 0);
      if (std::find(done.begin(), done.end(), ring) != done.end())
        continue;
      done.emplace_back(ring);
      edm::LogVerbatim("HGCalGeom") << "Corners for " << hid;

      const auto cor = geom->getNewCorners(id);
      std::ostringstream st1;
      for (unsigned int k = 0; k < cor.size(); ++k)
        st1 << " [" << k << "] " << std::setprecision(4) << cor[k];
      edm::LogVerbatim("HGCalGeom") << st1.str();

      double r = cor[0].perp();
      auto itr = layerMap.find(layer);
      if (itr == layerMap.end()) {
        layerInfo info;
        info.ringMin = info.ringMax = ring.first;
        info.rMin = info.rMax = r;
        layerMap[layer] = info;
      } else {
        layerInfo info = itr->second;
        if (info.ringMin > ring.first) {
          info.ringMin = ring.first;
          info.rMin = r;
        } else if (info.ringMax < ring.first) {
          info.ringMax = ring.first;
          info.rMax = r;
        }
        layerMap[layer] = info;
      }
    }
  }
  edm::LogVerbatim("HGCalGeom") << "\n\nSummary of " << layerMap.size() << " Scintillator layers";
  for (auto itr = layerMap.begin(); itr != layerMap.end(); ++itr)
    edm::LogVerbatim("HGCalGeom") << "Layer " << itr->first << " lowest Ring " << (itr->second).ringMin << ":"
                                  << (itr->second).rMin << " largest Ring " << (itr->second).ringMax << ":"
                                  << (itr->second).rMax;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalValidScintTest);
