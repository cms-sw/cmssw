/**
 * Author: Shahram Rahatlou, University of Rome & INFN
 * Created: 22 Feb 2006
 * $Id: $
 **/

#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"

EcalGainRatios::EcalGainRatios() {
}

EcalGainRatios::~EcalGainRatios() {

}

void
EcalGainRatios::setValue(const uint32_t& id, const EcalMGPAGainRatio & value) {
  map_[id] = value;
}

