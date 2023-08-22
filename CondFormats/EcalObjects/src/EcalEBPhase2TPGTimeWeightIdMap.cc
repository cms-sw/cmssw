#include "CondFormats/EcalObjects/interface/EcalEBPhase2TPGTimeWeightIdMap.h"

EcalEBPhase2TPGTimeWeightIdMap::EcalEBPhase2TPGTimeWeightIdMap() {}

EcalEBPhase2TPGTimeWeightIdMap::~EcalEBPhase2TPGTimeWeightIdMap() {}

void EcalEBPhase2TPGTimeWeightIdMap::setValue(const uint32_t& id, const EcalEBPhase2TPGTimeWeights& value) { map_[id] = value; }
