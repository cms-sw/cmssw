/**
 * Author: Paolo Meridiani
 * Created: 14 Nov 2006
 * $Id: EcalPTMTemperatures.cc,v 1.1 2006/11/16 18:19:45 meridian Exp $
 **/

#include "CondFormats/EcalObjects/interface/EcalPTMTemperatures.h"

EcalPTMTemperatures::EcalPTMTemperatures() {
}

EcalPTMTemperatures::~EcalPTMTemperatures() {

}

void
EcalPTMTemperatures::setValue(const uint32_t& id, const float& value) {
  map_[id] = value;
}

