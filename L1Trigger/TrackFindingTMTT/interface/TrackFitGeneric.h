#ifndef L1Trigger_TrackFindingTMTT_TrackFitGeneric_h
#define L1Trigger_TrackFindingTMTT_TrackFitGeneric_h

///=== This is the base class for all the track fit algorithms

///=== Written by: Alexander D. Morton and Sioni Summers

#include "L1Trigger/TrackFindingTMTT/interface/L1fittedTrack.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1track3D.h"

#include <vector>
#include <utility>

namespace tmtt {

  class Settings;

  class TrackFitGeneric {
  public:
    // Set configuration parameters.
    TrackFitGeneric(const Settings* settings, const std::string& fitterName = "")
        : settings_(settings), fitterName_(fitterName) {}

    virtual ~TrackFitGeneric() = default;

    // Fit a track candidate obtained from the Hough Transform.
    virtual L1fittedTrack fit(const L1track3D& l1track3D) { return L1fittedTrack(); }

  protected:
    // Configuration parameters
    const Settings* settings_;
    const std::string fitterName_;
  };

}  // namespace tmtt

#endif
