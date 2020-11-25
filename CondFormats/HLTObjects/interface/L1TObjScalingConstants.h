#ifndef HLTObjects_L1TObjScalingConstants_h
#define HLTObjects_L1TObjScalingConstants_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>

class L1TObjScalingConstants {
public:
  struct Scaling {
    float m_constant;
    float m_linear;
    float m_quadratic;
    COND_SERIALIZABLE;
  };

  std::vector<Scaling> m_constants;
  COND_SERIALIZABLE;
};

#endif  // HLTObjects_L1TObjScalingConstants_h
