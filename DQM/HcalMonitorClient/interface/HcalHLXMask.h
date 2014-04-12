#include "DQM/HcalMonitorClient/interface/HcalDQMChannelQuality.h"

class HcalHLXMask{

 public:
  char* position;
  unsigned int crateId;
  unsigned int slotId;
  unsigned int occMask;
  unsigned int lhcMask;
  unsigned int sumEtMask;
  void setMaskFromDQMChannelQuality(const HcalDQMChannelQuality::Item item[24]);

};


