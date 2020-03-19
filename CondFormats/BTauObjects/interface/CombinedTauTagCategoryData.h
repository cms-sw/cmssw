#ifndef CombinedTauTagCategoryData_H
#define CombinedTauTagCategoryData_H

#include "CondFormats/Serialization/interface/Serializable.h"

struct CombinedTauTagCategoryData {
  int truthmatched1orfake0candidates, theTagVar, signaltks_n;
  float EtMin, EtMax;

  COND_SERIALIZABLE;
};
#endif
