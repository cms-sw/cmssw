#ifndef CombinedTauTagCalibration_h
#define CombinedTauTagCalibration_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/BTauObjects/interface/CombinedTauTagCategoryData.h"
#include "CondFormats/BTauObjects/interface/CalibratedHistogram.h"
#include <vector>

struct CombinedTauTagCalibration {
  struct Entry {
    CombinedTauTagCategoryData category;
    CalibratedHistogram histogram;

    COND_SERIALIZABLE;
  };

  std::vector<Entry> data;

  COND_SERIALIZABLE;
};
#endif  //CombinedTauTagCalibration_h
