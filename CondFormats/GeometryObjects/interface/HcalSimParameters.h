#ifndef CondFormats_GeometryObjects_HcalSimParameters_h
#define CondFormats_GeometryObjects_HcalSimParameters_h

#include "CondFormats/Serialization/interface/Serializable.h"

class HcalSimParameters {
public:
  HcalSimParameters(void) {}
  ~HcalSimParameters(void) {}

  std::vector<double> attenuationLength_;
  std::vector<int> lambdaLimits_;
  std::vector<double> shortFiberLength_;
  std::vector<double> longFiberLength_;

  std::vector<int> pmtRight_;
  std::vector<int> pmtFiberRight_;
  std::vector<int> pmtLeft_;
  std::vector<int> pmtFiberLeft_;

  COND_SERIALIZABLE;
};

#endif
