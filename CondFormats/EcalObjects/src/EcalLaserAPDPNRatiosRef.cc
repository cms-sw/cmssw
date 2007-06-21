/**
 * Author: Vladlen Timciuc, Caltech, Pasadena, USA
 * Created: 15 May 2007
 * $Id: EcalLaserAPDPNRetiosRef.cc,v 1.1 2007/05/16 11:53:00 vladlen Exp $
 **/

#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatiosRef.h"

EcalLaserAPDPNRatiosRef::EcalLaserAPDPNRatiosRef() {
}

EcalLaserAPDPNRatiosRef::~EcalLaserAPDPNRatiosRef() {

}

void
EcalLaserAPDPNRatiosRef::setValue(const uint32_t& id, const EcalLaserAPDPNref & value) {
  map_[id] = value;
}

