#ifndef GUARD_HcalFlagHFDigiTimeParam
#define GUARD_HcalFlagHFDigiTimeParam

/**
\class HcalFlagHFDigiTimeParam
\author Jeff Temple

DB object to store HF DigiTime flag parameters
Flag parameters stored are:

 HFDigiTimeParam needs the following arguments:
 *  HFdigiflagFirstSample, HFdigiflagSamplesToAdd, HFdigiflagExpectedPeak -- values used to set the digi sections used for forming TS(peak)/NTS ratio

 *  HFdigiflagMinEthreshold -- minimum energy threshold needed before a rechit may be considered for flagging
 
 *  HFdigiflagCoefficients:  vector of doubles that define cut contour.  Current contour is [0]-exp([1]+[2]*E), but I think we want this to be a vector, so that we can allow for [0]-exp([1]+[2]*E+[3]*E*E), or [0], etc.

**/

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <cstdint>

class HcalFlagHFDigiTimeParam {
  // Default constructor sets parameters according to 6-TS digis
  // (Is there a way to pass in default vector arguments directly, rather than by using push_back?)
public:
  HcalFlagHFDigiTimeParam()
      : mId(0),
        mHFdigiflagFirstSample(1),
        mHFdigiflagSamplesToAdd(3),
        mHFdigiflagExpectedPeak(2),
        mHFdigiflagMinEthreshold(40) {
    mHFdigiflagCoefficients.clear();
    mHFdigiflagCoefficients.push_back(0.93);
    mHFdigiflagCoefficients.push_back(-0.38275);
    mHFdigiflagCoefficients.push_back(-0.012667);
  }

  // Set parameters based on input
  HcalFlagHFDigiTimeParam(unsigned long fId,
                          unsigned int fFirstSample,
                          unsigned int fSamplesToAdd,
                          unsigned int fExpectedPeak,
                          double fminEThreshold,
                          const std::vector<double>& fcoef)
      : mId(fId),
        mHFdigiflagFirstSample(fFirstSample),
        mHFdigiflagSamplesToAdd(fSamplesToAdd),
        mHFdigiflagExpectedPeak(fExpectedPeak),
        mHFdigiflagMinEthreshold(fminEThreshold),
        mHFdigiflagCoefficients(fcoef) {}

  uint32_t rawId() const { return mId; }

  // Return flag parameter values
  uint32_t HFdigiflagFirstSample() const { return mHFdigiflagFirstSample; }
  uint32_t HFdigiflagSamplesToAdd() const { return mHFdigiflagSamplesToAdd; }
  uint32_t HFdigiflagExpectedPeak() const { return mHFdigiflagExpectedPeak; }
  double HFdigiflagMinEThreshold() const { return mHFdigiflagMinEthreshold; }
  std::vector<double> HFdigiflagCoefficients() const { return mHFdigiflagCoefficients; }

private:
  uint32_t mId;                      // detector ID
  uint32_t mHFdigiflagFirstSample;   // first sample used in NTS calculation
  uint32_t mHFdigiflagSamplesToAdd;  // # of sampels to use in NTS calculation
  uint32_t mHFdigiflagExpectedPeak;  // expected peak position; used for calculating TS(peak)
  double mHFdigiflagMinEthreshold;   // minimum energy for flagged rechit
  std::vector<double>
      mHFdigiflagCoefficients;  // coefficients used to parameterize TS(peak)/NTS threshold:  [0]-exp([1]+[2]*E+....)

  COND_SERIALIZABLE;
};

#endif
