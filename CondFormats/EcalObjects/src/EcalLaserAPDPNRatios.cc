/**
 * Author: Vladlen Timciuc, Caltech
 * Created: 14 May 2007
 * $Id: EcalLaserAPDPNRatios.cc, 1.1 2007/05/16 11:45:00 Vladlen Exp $
 **/

#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatios.h"

EcalLaserAPDPNRatios::EcalLaserAPDPNRatios() {
}

EcalLaserAPDPNRatios::~EcalLaserAPDPNRatios() {

}

void 
EcalLaserAPDPNRatios::setValue(const uint32_t& id, const EcalLaserAPDPNpair& value) {
  laser_map[id] = value;
}

void 
EcalLaserAPDPNRatios::setTime(const int& id, const EcalLaserTimeStamp& value) {
  time_map[id] = value;
}


