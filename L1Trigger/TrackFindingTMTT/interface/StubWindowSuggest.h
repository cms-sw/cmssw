#ifndef L1Trigger_TrackFindingTMTT_StubWindowsSuggest_h
#define L1Trigger_TrackFindingTMTT_StubWindowsSuggest_h

#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"

#include <vector>

class TrackerTopology;

namespace tmtt {

  class Stub;

  /** 
 * ========================================================================================================
 *  This provides recommendations to CMS for the stub window sizes to be used in the FE electronics.
 *  It prints the output as a python configuration file in the form of
 *  L1Trigger/TrackTrigger/python/TTStubAlgorithmRegister_cfi.py .
 *
 *  The recommendations are based on the TMTT method of using the stub bend. Whilst they give 
 *  high efficiency, they do not take into account the requirement to limit the FE electronics band-width,
 *  so tighter cuts may be needed in reality.
 * ========================================================================================================
 */

  class StubWindowSuggest {
  public:
    // Initialize (for use with TMTT).
    StubWindowSuggest(const Settings* settings) : settings_(settings), ptMin_(settings->houghMinPt()) {}

    ~StubWindowSuggest() {}

    // Analyse stub window required for this stub.
    void process(const TrackerTopology* trackerTopo, const Stub* stub);

    // Print results (should be done in endJob();
    void printResults() const;

  private:
    // Update stored stub window size with this stub.
    void updateStoredWindow(const TrackerTopology* trackerTopo, const Stub* stub, double bendWind);

  private:
    // Configuration parameters.
    const Settings* settings_;
    const float ptMin_;

    // Stub window sizes as encoded in L1Trigger/TrackTrigger/interface/TTStubAlgorithm_official.h
    std::vector<double> barrelCut_;
    std::vector<std::vector<double> > ringCut_;
    std::vector<std::vector<double> > tiltedCut_;
    std::vector<double> barrelNTilt_;
  };

}  // namespace tmtt

#endif
