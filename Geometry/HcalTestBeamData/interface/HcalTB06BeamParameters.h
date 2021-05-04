#ifndef HcalTestBeamData_HcalTB06BeamParameters_h
#define HcalTestBeamData_HcalTB06BeamParameters_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <string>
#include <vector>

class HcalTB06BeamParameters {
public:
  HcalTB06BeamParameters() = default;

  std::vector<std::string> wchambers_;
  std::string material_;

  COND_SERIALIZABLE;
};

#endif
