#ifndef HcalTimeCorr_h
#define HcalTimeCorr_h

/*
\class HcalTimeCorr
\author Radek Ofierzynski
contains one time correction factor value + corresponding DetId
*/

class HcalTimeCorr
{
 public:
  HcalTimeCorr(): mId(0), mValue(0) {}
  HcalTimeCorr(unsigned long fid, float value): mId(fid), mValue(value) {}

  uint32_t rawId() const {return mId;}

  float getValue() const {return mValue;}

 private:
  uint32_t mId;
  float mValue;
};

#endif
