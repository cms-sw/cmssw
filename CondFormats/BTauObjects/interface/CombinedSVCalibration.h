#ifndef CombinedSVCalibration_h
#define CombinedSVCalibration_h

#include "CondFormats/BTauObjects/interface/CombinedSVCategoryData.h"
#include "CondFormats/BTauObjects/interface/CalibratedHistogram.h"
#include <vector>

struct CombinedSVCalibration
{
  struct Entry
  {
    CombinedSVCategoryData category;
    CalibratedHistogram histogram;
  };

 std::vector<Entry> data;
  
};

#endif //CombinedSVCalibration_h


