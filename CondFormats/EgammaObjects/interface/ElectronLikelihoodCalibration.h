#ifndef ElectronLikelihoodCalibration_h
#define ElectronLikelihoodCalibration_h

#include "CondFormats/EgammaObjects/interface/ElectronLikelihoodCategoryData.h"
#include "CondFormats/PhysicsToolsObjects/interface/Histogram.h"
#include <vector>

struct ElectronLikelihoodCalibration 
{

  struct Entry {
    
    ElectronLikelihoodCategoryData category;
    PhysicsTools::Calibration::HistogramF histogram;

  };

  std::vector<Entry> data;
  
};

#endif //ElectronLikelihoodCalibration_h
