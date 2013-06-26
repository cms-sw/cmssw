#ifndef CombinedTauTagCalibration_h
#define CombinedTauTagCalibration_h

#include "CondFormats/BTauObjects/interface/CombinedTauTagCategoryData.h"
#include "CondFormats/BTauObjects/interface/CalibratedHistogram.h"
#include <vector>

struct CombinedTauTagCalibration
{
  struct Entry
  {
   CombinedTauTagCategoryData category;
   CalibratedHistogram histogram;
  };

 std::vector<Entry> data;
  
};
#endif //CombinedTauTagCalibration_h


