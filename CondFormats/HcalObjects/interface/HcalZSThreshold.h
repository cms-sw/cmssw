#ifndef HcalZSThreshold_h
#define HcalZSThreshold_h

/*
\class HcalZSThreshold
\author Radek Ofierzynski
contains one threshold + corresponding DetId
*/

#include <boost/cstdint.hpp>

class HcalZSThreshold
{
 public:
  HcalZSThreshold(): mId(0), mLevel(0) {}
  HcalZSThreshold(unsigned long fid, int level): mId(fid), mLevel(level) {}

  uint32_t rawId() const {return mId;}

  int getValue() const {return mLevel;}

 private:
  uint32_t mId;
  int mLevel;
};

#endif
