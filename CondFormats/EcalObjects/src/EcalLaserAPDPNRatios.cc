/**
 * Author: Vladlen Timciuc, Caltech
 * Created: 14 May 2007
 * $Id: EcalLaserAPDPNRatios.cc,v 1.1 2007/06/21 13:56:37 meridian Exp $
 **/

#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatios.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

EcalLaserAPDPNRatios::EcalLaserAPDPNRatios() :
laser_map(EBDetId::MAX_HASH + EEDetId::MAX_HASH + 2),
time_map(92) // FIXME
{
}

EcalLaserAPDPNRatios::~EcalLaserAPDPNRatios() {

}


