#ifndef ElectronLikelihoodCalibration_h
#define ElectronLikelihoodCalibration_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/EgammaObjects/interface/ElectronLikelihoodCategoryData.h"
#include "CondFormats/PhysicsToolsObjects/interface/Histogram.h"

#include <vector>
#include <atomic>

struct ElectronLikelihoodCalibration {
  struct Entry {
    ElectronLikelihoodCategoryData category;
    PhysicsTools::Calibration::HistogramF histogram;

    COND_SERIALIZABLE;
  };

  std::vector<Entry> data;

  COND_SERIALIZABLE;
};

#endif  //ElectronLikelihoodCalibration_h
