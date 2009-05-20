#ifndef HcalPFCorr_h
#define HcalPFCorr_h

/*
\class HcalPFCorr
\author Radek Ofierzynski
contains one PF correction factor value + corresponding DetId
*/

class HcalPFCorr
{
 public:
  HcalPFCorr(): mId(0), mValue(0) {}
  HcalPFCorr(unsigned long fid, float value): mId(fid), mValue(value) {}

  uint32_t rawId() const {return mId;}

  float getValue() const {return mValue;}

 private:
  uint32_t mId;
  float mValue;
};

#endif
