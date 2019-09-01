#ifndef TrackProbabilityCalibration_h
#define TrackProbabilityCalibration_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/BTauObjects/interface/TrackProbabilityCategoryData.h"
#include "CondFormats/PhysicsToolsObjects/interface/Histogram.h"

#include <vector>

struct TrackProbabilityCalibration {
  struct Entry {
    TrackProbabilityCategoryData category;
    PhysicsTools::Calibration::HistogramF histogram;

    COND_SERIALIZABLE;
  };

  std::vector<Entry> data;

  COND_SERIALIZABLE;
};

#endif  //TrackProbabilityCalibration_h
