#include "L1Trigger/TrackFindingTMTT/interface/StubFEWindows.h"
#include "L1Trigger/TrackFindingTMTT/interface/TrackerModule.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"

#include <algorithm>
#include <utility>

using namespace std;

namespace tmtt {

  //=== Initialize stub window sizes from TTStubProducer cfg.

  StubFEWindows::StubFEWindows(const edm::ParameterSet& pSetStubAlgo) {
    numTiltedLayerRings_ = pSetStubAlgo.getParameter<vector<double>>("NTiltedRings");
    windowSizeBarrelLayers_ = pSetStubAlgo.getParameter<vector<double>>("BarrelCut");
    const auto& pSetTiltedLayer = pSetStubAlgo.getParameter<vector<edm::ParameterSet>>("TiltedBarrelCutSet");
    const auto& pSetEncapDisks = pSetStubAlgo.getParameter<vector<edm::ParameterSet>>("EndcapCutSet");
    windowSizeTiltedLayersRings_.reserve(pSetTiltedLayer.size());
    for (const auto& pSet : pSetTiltedLayer) {
      windowSizeTiltedLayersRings_.emplace_back(pSet.getParameter<vector<double>>("TiltedCut"));
    }
    windowSizeEndcapDisksRings_.reserve(pSetEncapDisks.size());
    for (const auto& pSet : pSetEncapDisks) {
      windowSizeEndcapDisksRings_.emplace_back(pSet.getParameter<vector<double>>("EndcapCut"));
    }
  }

  //=== Set all FE stub bend windows to zero.

  void StubFEWindows::setZero() {
    std::fill(windowSizeBarrelLayers_.begin(), windowSizeBarrelLayers_.end(), 0.);
    for (auto& x : windowSizeEndcapDisksRings_)
      std::fill(x.begin(), x.end(), 0.);
    for (auto& y : windowSizeTiltedLayersRings_)
      std::fill(y.begin(), y.end(), 0.);
  }

  //=== Const/non-const access to element of array giving window size for specific module.

  const double* StubFEWindows::storedWindowSize(const TrackerTopology* trackerTopo, const DetId& detId) const {
    // Code accessing geometry inspired by L1Trigger/TrackTrigger/src/TTStubAlgorithm_official.cc

    const double* storedHalfWindow = nullptr;
    if (detId.subdetId() == StripSubdetector::TOB) {
      unsigned int layer = trackerTopo->layer(detId);
      unsigned int ladder = trackerTopo->tobRod(detId);
      int type = 2 * trackerTopo->tobSide(detId) - 3;  // -1 for tilted-, 1 for tilted+, 3 for flat
      double corr = 0;

      if (type != TrackerModule::BarrelModuleType::flat) {
        // Tilted barrel
        corr = (numTiltedLayerRings_.at(layer) + 1) / 2.;
        // Corrected ring number, between 0 and barrelNTilt.at(layer), in ascending |z|
        ladder = corr - (corr - ladder) * type;
        storedHalfWindow = &(windowSizeTiltedLayersRings_.at(layer).at(ladder));
      } else {
        // Flat barrel
        storedHalfWindow = &(windowSizeBarrelLayers_.at(layer));
      }

    } else if (detId.subdetId() == StripSubdetector::TID) {
      // Endcap
      unsigned int wheel = trackerTopo->tidWheel(detId);
      unsigned int ring = trackerTopo->tidRing(detId);
      storedHalfWindow = &(windowSizeEndcapDisksRings_.at(wheel).at(ring));
    }
    return storedHalfWindow;
  }

  double* StubFEWindows::storedWindowSize(const TrackerTopology* trackerTopo, const DetId& detId) {
    // Code accessing geometry inspired by L1Trigger/TrackTrigger/src/TTStubAlgorithm_official.cc
    // Non-const version of operator, without needing to duplicate code.
    // (Scott Meyers trick).
    return const_cast<double*>(std::as_const(*this).storedWindowSize(trackerTopo, detId));
  }
}  // namespace tmtt
