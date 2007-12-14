#include "DQM/HcalMonitorClient/interface/HcalHLXMask.h"

void HcalHLXMask::setMaskFromDQMChannelQuality(const HcalDQMChannelQuality::Item item[24]){

  //just a stub... for now we set the masks to 0
  //that is, for now, we don't mask anything

  occMask = 0;
  lhcMask = 0;
  sumEtMask = 0;
}

