#include "Geometry/HcalCommonData/interface/HcalTopologyMode.h"

template <>
StringToEnumParser<HcalTopologyMode::Mode>::StringToEnumParser() {
  enumMap["HcalTopologyMode::LHC"] = HcalTopologyMode::LHC;
  enumMap["HcalTopologyMode::H2"] = HcalTopologyMode::H2;
  enumMap["HcalTopologyMode::SLHC"] = HcalTopologyMode::SLHC;
  enumMap["HcalTopologyMode::H2HE"] = HcalTopologyMode::H2HE;
  enumMap["HcalTopologyMode::Run3"] = HcalTopologyMode::Run3;
  enumMap["HcalTopologyMode::Run4"] = HcalTopologyMode::Run4;
  enumMap["HcalTopologyMode::Run2A"] = HcalTopologyMode::Run2A;
  enumMap["HcalTopologyMode::Run2B"] = HcalTopologyMode::Run2B;
  enumMap["HcalTopologyMode::Run2C"] = HcalTopologyMode::Run2C;
}

template <>
StringToEnumParser<HcalTopologyMode::TriggerMode>::StringToEnumParser() {
  enumMap["HcalTopologyMode::TriggerMode_2009"] = HcalTopologyMode::TriggerMode_2009;
  enumMap["HcalTopologyMode::TriggerMode_2016"] = HcalTopologyMode::TriggerMode_2016;
  enumMap["HcalTopologyMode::TriggerMode_2018legacy"] = HcalTopologyMode::TriggerMode_2018legacy;
  enumMap["HcalTopologyMode::TriggerMode_2017"] = HcalTopologyMode::TriggerMode_2017;
  enumMap["HcalTopologyMode::TriggerMode_2017plan1"] = HcalTopologyMode::TriggerMode_2017plan1;
  enumMap["HcalTopologyMode::TriggerMode_2018"] = HcalTopologyMode::TriggerMode_2018;
  enumMap["HcalTopologyMode::TriggerMode_2021"] = HcalTopologyMode::TriggerMode_2021;
}
