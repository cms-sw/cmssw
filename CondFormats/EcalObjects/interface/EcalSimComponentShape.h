#ifndef CondFormats_EcalObjects_EcalSimComponentShape_hh
#define CondFormats_EcalObjects_EcalSimComponentShape_hh

#include "CondFormats/Serialization/interface/Serializable.h"
#include <vector>

class EcalSimComponentShape {
public:
  EcalSimComponentShape() = default;

  std::vector<std::vector<float> > barrel_shapes;  // there is no need to getters/setters, just access data directly

  double barrel_thresh;
  float time_interval;  // time interval of the shape

  COND_SERIALIZABLE;
};
#endif
