#ifndef MuScleFitDBobject_h
#define MuScleFitDBobject_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>

struct MuScleFitDBobject {
  std::vector<int> identifiers;
  std::vector<double> parameters;
  std::vector<double> fitQuality;

  COND_SERIALIZABLE;
};

#endif  // MuScleFitDBobject
