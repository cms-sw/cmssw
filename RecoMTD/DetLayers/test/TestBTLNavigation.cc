#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "Geometry/MTDGeometryBuilder/interface/MTDTopology.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"

#include "RecoMTD/DetLayers/interface/MTDDetLayerGeometry.h"
#include "RecoMTD/Records/interface/MTDRecoGeometryRecord.h"

#include "RecoMTD/DetLayers/interface/MTDTrayBarrelLayer.h"
#include "RecoMTD/DetLayers/interface/MTDDetTray.h"

#include <DataFormats/ForwardDetId/interface/BTLDetId.h>

#include <sstream>

#include "DataFormats/Math/interface/Rounding.h"

using namespace std;
using namespace edm;
using namespace cms_rounding;

class TestBTLNavigation : public global::EDAnalyzer<> {
public:
  TestBTLNavigation(const ParameterSet& pset);

  void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;

private:
  inline std::string fround(const double in, const size_t prec) const {
    std::stringstream ss;
    ss << std::setprecision(prec) << std::fixed << std::setw(14) << roundIfNear0(in);
    return ss.str();
  }

  inline std::string fvecround(const GlobalPoint vecin, const size_t prec) const {
    std::stringstream ss;
    ss << std::setprecision(prec) << std::fixed << std::setw(14) << roundVecIfNear0(vecin);
    return ss.str();
  }

  const edm::ESInputTag tag_;
  edm::ESGetToken<MTDDetLayerGeometry, MTDRecoGeometryRecord> geomToken_;
  edm::ESGetToken<MTDTopology, MTDTopologyRcd> topoToken_;
};

TestBTLNavigation::TestBTLNavigation(const ParameterSet& iConfig) : tag_(edm::ESInputTag{"", ""}) {
  geomToken_ = esConsumes<MTDDetLayerGeometry, MTDRecoGeometryRecord>(tag_);
  topoToken_ = esConsumes<MTDTopology, MTDTopologyRcd>(tag_);
}

void TestBTLNavigation::analyze(edm::StreamID, edm::Event const&, edm::EventSetup const& es) const {
  auto geo = es.getTransientHandle(geomToken_);
  auto topo = es.getTransientHandle(topoToken_);

  const vector<const DetLayer*>& layers = geo->allBTLLayers();

  // dump of BTL layers structure

  LogVerbatim("MTDLayerDumpFull") << "\n\nTest of BTL navigation \n\n";
  LogVerbatim("MTDLayerDump") << "\n\nTest of BTL navigation \n\n";

  for (const auto& ilay : layers) {
    const MTDTrayBarrelLayer* layer = static_cast<const MTDTrayBarrelLayer*>(ilay);

    LogVerbatim("MTDLayerDumpFull") << std::fixed << "\nBTL layer " << std::setw(4) << layer->subDetector()
                                    << " at z = " << fround(layer->surface().position().z(), 4)
                                    << " rods = " << std::setw(14) << layer->rods().size()
                                    << " dets = " << std::setw(14) << layer->basicComponents().size();
    LogVerbatim("MTDLayerDump") << std::fixed << "\nBTL layer " << std::setw(4) << layer->subDetector()
                                << " at z = " << fround(layer->surface().position().z(), 2)
                                << " rods = " << std::setw(14) << layer->rods().size() << " dets = " << std::setw(14)
                                << layer->basicComponents().size();

    unsigned int irodInd(0);
    unsigned int imodInd(0);
    for (const auto& irod : layer->rods()) {
      irodInd++;
      LogVerbatim("MTDLayerDumpFull") << std::fixed << "\nTray " << std::setw(4) << irodInd << "\n"
                                      << " MTDDetTray at " << std::fixed << std::setprecision(3)
                                      << roundVecIfNear0(irod->specificSurface().position()) << std::endl
                                      << " L/W/T   : " << fround(irod->specificSurface().bounds().length(), 2) << " / "
                                      << fround(irod->specificSurface().bounds().width(), 2) << " / "
                                      << fround(irod->specificSurface().bounds().thickness(), 2)
                                      << " normal phi = " << fround(irod->specificSurface().normalVector().phi(), 2)
                                      << std::endl;
      LogVerbatim("MTDLayerDump") << std::fixed << "\nTray " << std::setw(4) << irodInd << "\n"
                                  << " MTDDetTray at " << std::fixed << std::setprecision(3)
                                  << roundVecIfNear0(irod->specificSurface().position()) << std::endl
                                  << " L/W/T   : " << fround(irod->specificSurface().bounds().length(), 2) << " / "
                                  << fround(irod->specificSurface().bounds().width(), 2) << " / "
                                  << fround(irod->specificSurface().bounds().thickness(), 2)
                                  << " normal phi = " << fround(irod->specificSurface().normalVector().phi(), 2)
                                  << std::endl;
      for (const auto& imod : irod->basicComponents()) {
        imodInd++;
        BTLDetId modId(imod->geographicalId().rawId());
        auto topoId = topo.product()->btlIndex(modId.rawId());
        auto topoDetId = topo.product()->btlidFromIndex(topoId.first, topoId.second);
        LogVerbatim("MTDLayerDumpFull") << std::fixed << std::setw(5) << imodInd << " BTLDetId " << modId.rawId()
                                        << " iphi/ieta = " << std::setw(4) << topoId.first << " / " << std::setw(4)
                                        << topoId.second << " side = " << std::setw(4) << modId.mtdSide()
                                        << " RU = " << std::setw(4) << modId.runit() << " mod = " << std::setw(4)
                                        << modId.module() << " pos = " << fvecround(imod->position(), 4);
        LogVerbatim("MTDLayerDump") << std::fixed << std::setw(5) << imodInd << " BTLDetId " << modId.rawId()
                                    << " iphi/ieta = " << std::setw(4) << topoId.first << " / " << std::setw(4)
                                    << topoId.second << " side = " << std::setw(4) << modId.mtdSide()
                                    << " RU = " << std::setw(4) << modId.runit() << " mod = " << std::setw(4)
                                    << modId.module() << " pos = " << fvecround(imod->position(), 2);
        if (topoDetId != modId.rawId()) {
          LogVerbatim("MTDLayerDumpFull")
              << "DIFFERENCE BtlDetId " << modId.rawId() << " not equal to MTDTopology " << topoDetId;
          LogVerbatim("MTDLayerDump") << "DIFFERENCE BtlDetId " << modId.rawId() << " not equal to MTDTopology "
                                      << topoDetId;
        }
        for (int iside = -1; iside <= 1; iside += 2) {
          size_t idetNew = topo.product()->phishiftBTL(modId.rawId(), iside);
          if (idetNew >= layer->basicComponents().size()) {
            LogVerbatim("MTDLayerDumpFull")
                << "...............phishift= " << std::fixed << std::setw(2) << iside << " out of range";
            LogVerbatim("MTDLayerDump") << "...............phishift= " << std::fixed << std::setw(2) << iside
                                        << " out of range";
          } else {
            BTLDetId newId(layer->basicComponents()[idetNew]->geographicalId().rawId());
            auto const& newTopoId = topo.product()->btlIndex(newId.rawId());
            LogVerbatim("MTDLayerDumpFull")
                << std::fixed << "...............phishift= "
                << " iphi/ieta = " << std::setw(4) << newTopoId.first << " / " << std::setw(4) << newTopoId.second
                << std::setw(4) << iside << " side = " << std::setw(4) << newId.mtdSide() << " RU = " << std::setw(4)
                << newId.runit() << " mod = " << std::setw(4) << newId.module()
                << " pos = " << fvecround(layer->basicComponents()[idetNew]->position(), 4);
            LogVerbatim("MTDLayerDump") << std::fixed << "...............phishift= "
                                        << " iphi/ieta = " << std::setw(4) << newTopoId.first << " / " << std::setw(4)
                                        << newTopoId.second << std::setw(4) << iside << " side = " << std::setw(4)
                                        << newId.mtdSide() << " RU = " << std::setw(4) << newId.runit()
                                        << " mod = " << std::setw(4) << newId.module()
                                        << " pos = " << fvecround(layer->basicComponents()[idetNew]->position(), 2);
          }
        }
        for (int iside = -1; iside <= 1; iside += 2) {
          auto idetNew = topo.product()->etashiftBTL(modId, iside);
          if (idetNew >= layer->basicComponents().size()) {
            LogVerbatim("MTDLayerDumpFull")
                << "...............etashift= " << std::fixed << std::setw(2) << iside << " out of range";
            LogVerbatim("MTDLayerDump") << "...............etashift= " << std::fixed << std::setw(2) << iside
                                        << " out of range";
          } else {
            BTLDetId newId(layer->basicComponents()[idetNew]->geographicalId().rawId());
            auto const& newTopoId = topo.product()->btlIndex(newId.rawId());
            LogVerbatim("MTDLayerDumpFull")
                << std::fixed << "...............etashift= "
                << " iphi/ieta = " << std::setw(4) << newTopoId.first << " / " << std::setw(4) << newTopoId.second
                << std::setw(4) << iside << " side = " << std::setw(4) << newId.mtdSide() << " RU = " << std::setw(4)
                << newId.runit() << " mod = " << std::setw(4) << newId.module()
                << " pos = " << fvecround(layer->basicComponents()[idetNew]->position(), 4);
            LogVerbatim("MTDLayerDump") << std::fixed << "...............etashift= "
                                        << " iphi/ieta = " << std::setw(4) << newTopoId.first << " / " << std::setw(4)
                                        << newTopoId.second << std::setw(4) << iside << " side = " << std::setw(4)
                                        << newId.mtdSide() << " RU = " << std::setw(4) << newId.runit()
                                        << " mod = " << std::setw(4) << newId.module()
                                        << " pos = " << fvecround(layer->basicComponents()[idetNew]->position(), 2);
          }
        }
      }
    }
  }
}

//define this as a plug-in
#include <FWCore/Framework/interface/MakerMacros.h>
DEFINE_FWK_MODULE(TestBTLNavigation);
