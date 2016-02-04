#ifndef EVENTFILTER_HCALRAWTODIGI_HCALTTPUNPACKER_H
#define EVENTFILTER_HCALRAWTODIGI_HCALTTPUNPACKER_H 1

#include "DataFormats/HcalDigi/interface/HcalTTPDigi.h"
#include "EventFilter/HcalRawToDigi/interface/HcalHTRData.h"

/** \class HcalTTPUnpacker
  *  
  * $Date: 2009/09/11 19:56:30 $
  * $Revision: 1.1 $
  * \author J. Mans - Minnesota
  */
class HcalTTPUnpacker {
public:
  bool unpack(const HcalHTRData& data, HcalTTPDigi& digi);

};
#endif
