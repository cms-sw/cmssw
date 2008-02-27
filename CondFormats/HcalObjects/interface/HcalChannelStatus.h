#ifndef HcalChannelStatus_h
#define HcalChannelStatus_h

/* 
\class HcalChannelStatus
\author Radek Ofierzynski
contains one channel status and corresponding DetId
*/

#include <boost/cstdint.hpp>

class HcalChannelStatus
{
 public:
 HcalChannelStatus(): mId(0), mStatus(0), empty(true) {}
 HcalChannelStatus(unsigned long fid, uint32_t status): mId(fid), mStatus(status), empty(false) {}

  void setDetId(unsigned long fid) {mId = fid; empty = false;}
  void setValue(uint32_t value) {mStatus = value; empty = false;}

 void setBit(unsigned int bitnumber) 
    {
      uint32_t statadd = 0x1<<(bitnumber-1);
      mStatus = mStatus|statadd;
    }
 void unsetBit(unsigned int bitnumber) 
    {
      uint32_t statadd = 0x1<<(bitnumber-1);
      statadd = ~statadd;
      mStatus = mStatus&statadd;
    }

 uint32_t rawId() const {return mId;}

 uint32_t getValue() const {return mStatus;}
 bool isEmpty() const {return empty;}


  private:
    uint32_t mId;
    uint32_t mStatus;
    bool empty;

};
#endif
