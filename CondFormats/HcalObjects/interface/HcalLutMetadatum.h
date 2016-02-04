#ifndef HcalLutMetadatum_h
#define HcalLutMetadatum_h

/*
\class HcalLutMetadatum
\author Gena Kukartsev
contains RecHit calibration, LUT granularity, output LUT threshold for one HCAL channel
*/

class HcalLutMetadatum
{
 public:
  HcalLutMetadatum(): mId(0), mRCalib(1.0) {}
    HcalLutMetadatum(unsigned long fid,
		     float rCalib,
		     uint8_t lutGranularity,
		     uint8_t outputLutThreshold):
      mId(fid), 
      mRCalib(rCalib),
      mLutGranularity(lutGranularity),
      mOutputLutThreshold(outputLutThreshold) {}

  uint32_t rawId() const {return mId;}

  float   getRCalib() const {return mRCalib;}
  uint8_t getLutGranularity() const {return mLutGranularity;}
  uint8_t getOutputLutThreshold() const {return mOutputLutThreshold;}

 private:
  uint32_t mId;
  float    mRCalib;
  uint8_t  mLutGranularity;
  uint8_t  mOutputLutThreshold;
};

#endif
