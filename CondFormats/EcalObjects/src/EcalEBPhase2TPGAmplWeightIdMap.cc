#include "CondFormats/EcalObjects/interface/EcalEBPhase2TPGAmplWeightIdMap.h"

EcalEBPhase2TPGAmplWeightIdMap::EcalEBPhase2TPGAmplWeightIdMap() {}

EcalEBPhase2TPGAmplWeightIdMap::~EcalEBPhase2TPGAmplWeightIdMap() {}

void EcalEBPhase2TPGAmplWeightIdMap::setValue(const uint32_t& id, const EcalEBPhase2TPGAmplWeights& value) {
  map_[id] = value;
}
