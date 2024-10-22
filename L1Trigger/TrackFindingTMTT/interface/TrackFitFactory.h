#ifndef L1Trigger_TrackFindingTMTT_TrackFitFactory_h
#define L1Trigger_TrackFindingTMTT_TrackFitFactory_h

///=== Create requested track fitter

#include "L1Trigger/TrackFindingTMTT/interface/L1fittedTrack.h"
#include "L1Trigger/TrackFindingTMTT/interface/L1track3D.h"
#include "L1Trigger/TrackFindingTMTT/interface/TrackFitGeneric.h"

#include <vector>
#include <utility>
#include <memory>

namespace tmtt {

  class Settings;

  namespace trackFitFactory {

    // Function to produce a fitter based on a std::string
    std::unique_ptr<TrackFitGeneric> create(const std::string& fitterName, const Settings* settings);

  }  // namespace trackFitFactory

}  // namespace tmtt

#endif
