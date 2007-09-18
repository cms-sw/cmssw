#ifndef ElectronLikelihoodCalibration_h
#define ElectronLikelihoodCalibration_h

#include "CondFormats/EgammaObjects/interface/ElectronLikelihoodCategoryData.h"
#include "CondFormats/BTauObjects/interface/CalibratedHistogram.h"
#include <vector>

struct ElectronLikelihoodCalibration 
{

  struct Entry {
    
    ElectronLikelihoodCategoryData category;
    CalibratedHistogram histogram;

  };

  std::vector<Entry> data;
  
};

#endif //ElectronLikelihoodCalibration_h
