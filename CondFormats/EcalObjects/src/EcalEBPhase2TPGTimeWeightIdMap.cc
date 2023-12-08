#include "CondFormats/EcalObjects/interface/EcalEBPhase2TPGTimeWeightIdMap.h"

void EcalEBPhase2TPGTimeWeightIdMap::setValue(const uint32_t& id, const EcalEBPhase2TPGTimeWeights& value) {
  map_[id] = value;
}
