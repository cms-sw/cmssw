#ifndef EVENTFILTER_HCALRAWTODIGI_HCALTTPUNPACKER_H
#define EVENTFILTER_HCALRAWTODIGI_HCALTTPUNPACKER_H 1

#include "DataFormats/HcalDigi/interface/HcalTTPDigi.h"
#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"

/** \class HcalTTPUnpacker
  *  
  * \author J. Mans - Minnesota
  */
class HcalTTPUnpacker {
public:
  bool unpack(const HcalHTRData& data, HcalTTPDigi& digi);
};
#endif
