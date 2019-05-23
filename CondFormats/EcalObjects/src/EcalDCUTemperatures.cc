/**
 * Author: Paolo Meridiani
 * Created: 14 Nov 2006
 * $Id: $
 **/

#include "CondFormats/EcalObjects/interface/EcalDCUTemperatures.h"

EcalDCUTemperatures::EcalDCUTemperatures() {}

EcalDCUTemperatures::~EcalDCUTemperatures() {}

void EcalDCUTemperatures::setValue(const uint32_t& id, const float& value) { map_[id] = value; }
