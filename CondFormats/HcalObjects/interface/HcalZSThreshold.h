#ifndef HcalZSThreshold_h
#define HcalZSThreshold_h

/*
\class HcalZSThreshold
\author Radek Ofierzynski
contains one threshold + corresponding DetId
*/

class HcalZSThreshold
{
 public:
 HcalZSThreshold(): mId(0), mLevel(0), empty(true) {}
 HcalZSThreshold(unsigned long fid, int level): mId(fid), mLevel(level), empty(false) {}

    // because of an oracle digestion problem with uint32_t 
    // use unsigned long long
  //  uint32_t rawId() const {return mId;}
  unsigned long long rawId() const {return mId;}
  int getValue() const {return mLevel;}
  bool isEmpty() const {return empty;}

 private:
  //  uint32_t mId;
  unsigned long long mId;
  int mLevel;
  bool empty;
};

#endif
