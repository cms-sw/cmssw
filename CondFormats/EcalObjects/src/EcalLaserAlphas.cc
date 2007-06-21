/**
 * Author: Vladlen Timciuc, Caltech, Pasadena, USA
 * Created: 15 May 2007
 * $Id: EcalLaserAlphas.cc,v 1.1 2007/05/16 12:06:00 vladlen Exp $
 **/

#include "CondFormats/EcalObjects/interface/EcalLaserAlphas.h"

EcalLaserAlphas::EcalLaserAlphas() {
}

EcalLaserAlphas::~EcalLaserAlphas() {

}

void
EcalLaserAlphas::setValue(const uint32_t& id, const EcalLaserAlpha & value) {
  map_[id] = value;
}

