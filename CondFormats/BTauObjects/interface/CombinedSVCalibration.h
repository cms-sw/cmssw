#ifndef CombinedSVCalibration_h
#define CombinedSVCalibration_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/BTauObjects/interface/CombinedSVCategoryData.h"
#include "CondFormats/BTauObjects/interface/CalibratedHistogram.h"
#include <vector>

struct CombinedSVCalibration {
  struct Entry {
    CombinedSVCategoryData category;
    CalibratedHistogram histogram;

    COND_SERIALIZABLE;
  };

  std::vector<Entry> data;

  COND_SERIALIZABLE;
};

#endif  //CombinedSVCalibration_h
