#include "Geometry/HcalCommonData/interface/HcalTopologyMode.h"

template<>
StringToEnumParser<HcalTopologyMode::Mode>::StringToEnumParser() {
  enumMap["HcalTopologyMode::LHC"] = HcalTopologyMode::LHC;
  enumMap["HcalTopologyMode::H2"] = HcalTopologyMode::H2;
  enumMap["HcalTopologyMode::SLHC"] = HcalTopologyMode::SLHC;
  enumMap["HcalTopologyMode::H2HE"] = HcalTopologyMode::H2HE;
}
