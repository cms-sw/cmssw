#include "CondFormats/EcalObjects/interface/EcalTPGOddWeightIdMap.h"

EcalTPGOddWeightIdMap::EcalTPGOddWeightIdMap() {}

EcalTPGOddWeightIdMap::~EcalTPGOddWeightIdMap() {}

void EcalTPGOddWeightIdMap::setValue(const uint32_t& id, const EcalTPGWeights& value) { map_[id] = value; }
