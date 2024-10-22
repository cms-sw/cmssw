/**
 * Author: Paolo Meridiani
 * Created: 14 Nov 2006
 * $Id: $
 **/

#include "CondFormats/EcalObjects/interface/EcalPTMTemperatures.h"

EcalPTMTemperatures::EcalPTMTemperatures() {}

EcalPTMTemperatures::~EcalPTMTemperatures() {}

void EcalPTMTemperatures::setValue(const uint32_t& id, const float& value) { map_[id] = value; }
