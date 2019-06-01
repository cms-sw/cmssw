#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainEBIdMap.h"

EcalTPGFineGrainEBIdMap::EcalTPGFineGrainEBIdMap() {}

EcalTPGFineGrainEBIdMap::~EcalTPGFineGrainEBIdMap() {}

void EcalTPGFineGrainEBIdMap::setValue(const uint32_t& id, const EcalTPGFineGrainConstEB& value) { map_[id] = value; }
