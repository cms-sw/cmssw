#ifndef Geometry_HcalCommonData_HcalTopologyMode_H
#define Geometry_HcalCommonData_HcalTopologyMode_H

#include "FWCore/Utilities/interface/Exception.h"
#include <map>
#include <string>
#include <algorithm>

template <typename T>
class StringToEnumParser {
  std::map<std::string, T> enumMap;

public:
  StringToEnumParser(void);

  T parseString(const std::string &value) {
    typename std::map<std::string, T>::const_iterator iValue = enumMap.find(value);
    if (iValue == enumMap.end())
      throw cms::Exception("Configuration") << "the value " << value << " is not defined.";

    return iValue->second;
  }
};

namespace HcalTopologyMode {
  enum Mode {
    LHC = 0,    // Legacy HCAL
    H2 = 1,     // H2 TB
    SLHC = 2,   // Attemptf HE to be used for HGCal
    H2HE = 3,   // H2 TB with includng HE
    Run3 = 4,   // Run3 with inclusionof ZDC
    Run4 = 5,   // Post LS3
    Run2A = 6,  // With extended channels for HF
    Run2B = 7,  // With extended channels for HE
    Run2C = 8   // With extended channels for HB
  };

  enum TriggerMode {
    TriggerMode_2009 = 0,        // HF is summed in 3x2 regions
    TriggerMode_2016 = 1,        // HF is summed in both 3x2 and 1x1 regions
    TriggerMode_2018legacy = 2,  // For the database, before 2017 and 2017plan1 was introduced
    TriggerMode_2017 = 3,        // HF upgraded to QIE10
    TriggerMode_2017plan1 = 4,   // HF upgraded to QIE10, 1 RBX of HE to QIE11
    TriggerMode_2018 = 5,        // HF upgraded to QIE10, HE to QIE11
    TriggerMode_2021 = 6         // HF upgraded to QIE10, HBHE to QIE11
  };
}  // namespace HcalTopologyMode

#endif  // Geometry_HcalCommonData_HcalTopologyMode_H
