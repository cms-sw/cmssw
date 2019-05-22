#include "CondFormats/EcalObjects/interface/EcalTPGWeightIdMap.h"

EcalTPGWeightIdMap::EcalTPGWeightIdMap() {}

EcalTPGWeightIdMap::~EcalTPGWeightIdMap() {}

void EcalTPGWeightIdMap::setValue(const uint32_t& id, const EcalTPGWeights& value) { map_[id] = value; }
