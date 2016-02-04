#ifndef HcalRespCorr_h
#define HcalRespCorr_h

/*
\class HcalRespCorr
\author Radek Ofierzynski
contains one response correction factor value + corresponding DetId
*/

class HcalRespCorr
{
 public:
  HcalRespCorr(): mId(0), mValue(0) {}
  HcalRespCorr(unsigned long fid, float value): mId(fid), mValue(value) {}

  uint32_t rawId() const {return mId;}

  float getValue() const {return mValue;}

 private:
  uint32_t mId;
  float mValue;
};

#endif
