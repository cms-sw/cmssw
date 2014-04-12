#ifndef TrackProbabilityCalibration_h
#define TrackProbabilityCalibration_h

#include "CondFormats/BTauObjects/interface/TrackProbabilityCategoryData.h"
#include "CondFormats/PhysicsToolsObjects/interface/Histogram.h"

#include <vector>

struct TrackProbabilityCalibration
{
  struct Entry
  {
   TrackProbabilityCategoryData category;
    PhysicsTools::Calibration::HistogramF histogram;
  };

 std::vector<Entry> data;
  
};

#endif //TrackProbabilityCalibration_h


