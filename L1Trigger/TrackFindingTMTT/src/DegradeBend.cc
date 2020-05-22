#include "L1Trigger/TrackFindingTMTT/interface/DegradeBend.h"
#include "L1Trigger/TrackFindingTMTT/interface/TrackerModule.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <map>
#include <set>
#include <utility>
#include <cmath>
#include <algorithm>
#include <iostream>

using namespace std;

namespace tmtt {

  //--- Given the original bend, flag indicating if this is a PS or 2S module, & detector identifier,
  //--- this returns the degraded stub bend, a boolean indicatng if stub bend was outside the assumed window
  //--- size programmed below, and an integer indicating how many values of the original bend
  //--- were grouped together into this single value of the degraded bend.

  void DegradeBend::degrade(float bend,
                            bool psModule,
                            const DetId& stDetId,
                            float windowFEnew,
                            float& degradedBend,
                            unsigned int& numInGroup) const {
    // Get degraded bend value.
    unsigned int windowHalfStrips;
    this->work(bend, psModule, stDetId, windowFEnew, degradedBend, numInGroup, windowHalfStrips);
  }

  //--- Does the actual work of degrading the bend.

  void DegradeBend::work(float bend,
                         bool psModule,
                         const DetId& stDetId,
                         float windowFEnew,
                         float& degradedBend,
                         unsigned int& numInGroup,
                         unsigned int& windowHalfStrips) const {
    // Calculate stub window size in half-strip units used to produce stubs.
    // Code accessing geometry inspired by L1Trigger/TrackTrigger/src/TTStubAlgorithm_official.cc

    const double* storedHalfWindow = sw_->storedWindowSize(theTrackerTopo_, stDetId);

    // Compare this with the possibly tighter window provided by the user, converting to half-strip units.
    const double window = std::min(*storedHalfWindow, double(windowFEnew));
    windowHalfStrips = (unsigned int)(2 * window);

    // Bend is measured with granularity of 0.5 strips.
    // Convert it to integer measured in half-strip units for this calculation!
    int b = std::round(2 * bend);

    if ((unsigned int)(std::abs(b)) <= windowHalfStrips) {
      // Call the official CMS bend encoding algorithm.
      degradedBend = stubAlgo_->degradeBend(psModule, windowHalfStrips, b);
    } else {
      // This should only happen for stubs subsequently rejected by the FE.
      numInGroup = 0;
      constexpr float rejectedStubBend = 99999.;
      degradedBend = rejectedStubBend;
    }
  }
}  // namespace tmtt
