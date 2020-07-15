#ifndef HcalTestBeamData_HcalTB02Parameters_h
#define HcalTestBeamData_HcalTB02Parameters_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <map>
#include <string>

class HcalTB02Parameters {
public:
  HcalTB02Parameters(const std::string& nam) : name_(nam) {}

  std::string name_;
  std::map<std::string, double> lengthMap_;

  COND_SERIALIZABLE;
};

#endif
