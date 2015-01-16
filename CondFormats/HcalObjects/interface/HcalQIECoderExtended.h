#ifndef HcalQIECoderExtended_h
#define HcalQIECoderExtended_h

/** 
\class HcalQIECoderExtended
\author  Clemencia Mora (CBPF)
Extended version of QIECoder to add QIEId: barcode and channel
$Author: cmora
$Date: 2015/01/13  $
$Revision: 1.0 $
*/

#include "CondFormats/Serialization/interface/Serializable.h"
#include "CondFormats/HcalObjects/interface/HcalQIECoder.h"

#include <boost/cstdint.hpp>

#include <vector>
#include <algorithm>

class HcalQIEShape;

class HcalQIECoderExtended: public HcalQIECoder {
 public:
  HcalQIECoderExtended (unsigned long fId = 0) : mId(fId), mQIEbarcode(0), mQIEchannel(0) { setQIEIndex(0);}

  int getQIEbarcode()const { return mQIEbarcode; }
  int getQIEchannel()const { return mQIEchannel; }  

  void setQIEId(int bar, int chan);


 private:
  uint32_t mId;
  int mQIEbarcode ;
  int mQIEchannel ;

 COND_SERIALIZABLE;
};

#endif
