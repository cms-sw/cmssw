/**
 * Author: Vladlen Timciuc, Caltech, Pasadena, USA
 * Created: 15 May 2007
 * $Id: EcalLaserAlphas.cc,v 1.1 2007/06/21 13:56:37 meridian Exp $
 **/

#include "CondFormats/EcalObjects/interface/EcalLaserAlphas.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

EcalLaserAlphas::EcalLaserAlphas() :
map_(EBDetId::MAX_HASH + EEDetId::MAX_HASH + 2)
{
}

EcalLaserAlphas::~EcalLaserAlphas() {

}

