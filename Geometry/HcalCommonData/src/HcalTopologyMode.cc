#include "Geometry/HcalCommonData/interface/HcalTopologyMode.h"

template<>
StringToEnumParser<HcalTopologyMode::Mode>::StringToEnumParser() {
  enumMap["HcalTopologyMode::LHC"] = HcalTopologyMode::LHC;
  enumMap["HcalTopologyMode::H2"] = HcalTopologyMode::H2;
  enumMap["HcalTopologyMode::SLHC"] = HcalTopologyMode::SLHC;
  enumMap["HcalTopologyMode::H2HE"] = HcalTopologyMode::H2HE;
}

template<>
StringToEnumParser<HcalTopologyMode::TriggerMode>::StringToEnumParser() {
  enumMap["HcalTopologyMode::tm_LHC_RCT"] = HcalTopologyMode::tm_LHC_RCT;
  enumMap["HcalTopologyMode::tm_LHC_RCT_and_1x1"] = HcalTopologyMode::tm_LHC_RCT_and_1x1;
  enumMap["HcalTopologyMode::tm_LHC_1x1"] = HcalTopologyMode::tm_LHC_1x1;
}
