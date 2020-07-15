#ifndef L1Trigger_TrackFindingTMTT_StubFEWindows_h
#define L1Trigger_TrackFindingTMTT_StubFEWindows_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <vector>

// Window sizes used by FE electronics to select stubs.

class TrackerTopology;

namespace tmtt {

  class StubFEWindows {
  public:
    // Initialize stub window sizes from TTStubProducer cfg.
    StubFEWindows(const edm::ParameterSet& pSetStubAlgo);

    // Set all FE stub bend windows to zero.
    void setZero();

    // Access window size arrays (const functions).
    const std::vector<double>& windowSizeBarrelLayers() const { return windowSizeBarrelLayers_; }
    const std::vector<std::vector<double> >& windowSizeEndcapDisksRings() const { return windowSizeEndcapDisksRings_; }
    const std::vector<std::vector<double> >& windowSizeTiltedLayersRings() const {
      return windowSizeTiltedLayersRings_;
    }

    // Access window size arrays (non-const functions).
    std::vector<double>& windowSizeBarrelLayers() { return windowSizeBarrelLayers_; }
    std::vector<std::vector<double> >& windowSizeEndcapDisksRings() { return windowSizeEndcapDisksRings_; }
    std::vector<std::vector<double> >& windowSizeTiltedLayersRings() { return windowSizeTiltedLayersRings_; }

    // Number of tilted barrel modules each half of each PS barrel layer.
    const std::vector<double>& numTiltedLayerRings() const { return numTiltedLayerRings_; }

    // Const/non-const access to element of array giving window size for specific module.
    const double* storedWindowSize(const TrackerTopology* trackerTopo, const DetId& detId) const;
    double* storedWindowSize(const TrackerTopology* trackerTopo, const DetId& detId);

  private:
    // Stub window sizes as encoded in L1Trigger/TrackTrigger/interface/TTStubAlgorithm_official.h
    std::vector<double> windowSizeBarrelLayers_;
    std::vector<std::vector<double> > windowSizeEndcapDisksRings_;
    std::vector<std::vector<double> > windowSizeTiltedLayersRings_;
    std::vector<double> numTiltedLayerRings_;
  };

}  // namespace tmtt

#endif
