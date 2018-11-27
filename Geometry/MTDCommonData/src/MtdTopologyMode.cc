#include "Geometry/MTDCommonData/interface/MTDTopologyMode.h"

template<>
MTDStringToEnumParser<MTDTopologyMode::Mode>::MTDStringToEnumParser() {
  enumMap["MTDTopologyMode::tile"] = MTDTopologyMode::tile;
  enumMap["MTDTopologyMode::bar"] = MTDTopologyMode::bar;
  enumMap["MTDTopologyMode::barzflat"] = MTDTopologyMode::barzflat;
}
