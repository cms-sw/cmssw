/** 
\class HcalQIEDataExtended
\author Clemencia Mora Herrera
POOL object to store QIE coder parameters for one channel
$Author: cmora
$Date: 2015/01/13 $
$Revision: 1.0 $
*/

#include <iostream>

#include "CondFormats/HcalObjects/interface/HcalQIEShape.h"
#include "CondFormats/HcalObjects/interface/HcalQIECoderExtended.h"


void HcalQIECoderExtended::setQIEId (int bar, int chan) {
  mQIEbarcode = bar;
  mQIEchannel = chan;
}

