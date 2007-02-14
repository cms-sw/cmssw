#ifndef TrackProbabilityCalibration_h
#define TrackProbabilityCalibration_h

#include "CondFormats/BTauObjects/interface/TrackProbabilityCategoryData.h"
#include "CondFormats/BTauObjects/interface/CalibratedHistogram.h"
#include <vector>

struct TrackProbabilityCalibration
{
  struct Entry
  {
   TrackProbabilityCategoryData category;
   CalibratedHistogram histogram;
  };

 std::vector<Entry> data;
  
};

#endif //TrackProbabilityCalibration_h


