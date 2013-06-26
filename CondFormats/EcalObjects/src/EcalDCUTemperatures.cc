/**
 * Author: Paolo Meridiani
 * Created: 14 Nov 2006
 * $Id: EcalDCUTemperatures.cc,v 1.1 2006/11/16 18:19:45 meridian Exp $
 **/

#include "CondFormats/EcalObjects/interface/EcalDCUTemperatures.h"

EcalDCUTemperatures::EcalDCUTemperatures() {
}

EcalDCUTemperatures::~EcalDCUTemperatures() {

}

void
EcalDCUTemperatures::setValue(const uint32_t& id, const float& value) {
  map_[id] = value;
}

