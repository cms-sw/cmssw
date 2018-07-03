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
  enumMap["HcalTopologyMode::TriggerMode_2009"] = HcalTopologyMode::TriggerMode_2009;
  enumMap["HcalTopologyMode::TriggerMode_2016"] = HcalTopologyMode::TriggerMode_2016;
  enumMap["HcalTopologyMode::TriggerMode_2017"] = HcalTopologyMode::TriggerMode_2017;
  enumMap["HcalTopologyMode::TriggerMode_2017plan1"] = HcalTopologyMode::TriggerMode_2017plan1;
  enumMap["HcalTopologyMode::TriggerMode_2018"] = HcalTopologyMode::TriggerMode_2018;
  enumMap["HcalTopologyMode::TriggerMode_2019"] = HcalTopologyMode::TriggerMode_2019;
}
