#ifndef CondFormats_EcalObjects_EcalSimPulseShape_hh
#define CondFormats_EcalObjects_EcalSimPulseShape_hh

#include "CondFormats/Serialization/interface/Serializable.h"
#include <vector>

class EcalSimPulseShape {
public:
  EcalSimPulseShape(){};
  ~EcalSimPulseShape(){};
  void setTimeInterval(float x) { time_interval = x; };
  float getTimeInterval() { return time_interval; };

  std::vector<double> barrel_shape;  // there is no need to getters/setters, just access data directly
  std::vector<double> endcap_shape;  // there is no need to getters/setters, just access data directly
  std::vector<double> apd_shape;     // there is no need to getters/setters, just access data directly

  double barrel_thresh;
  double endcap_thresh;
  double apd_thresh;
  float time_interval;  // time interval of the shape

  COND_SERIALIZABLE;
};
#endif
