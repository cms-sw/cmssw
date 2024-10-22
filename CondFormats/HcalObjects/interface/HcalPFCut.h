#ifndef CondFormats_HcalObjects_HcalPFCut_h
#define CondFormats_HcalObjects_HcalPFCut_h

#include <cstdint>

#include "CondFormats/Serialization/interface/Serializable.h"

class HcalPFCut {
public:
  inline HcalPFCut() : mId_(0), noiseThresh_(0.f), seedThresh_(0.f) {}

  inline HcalPFCut(unsigned long fId, float noiseThresh, float seedThresh)
      : mId_(fId), noiseThresh_(noiseThresh), seedThresh_(seedThresh) {}

  inline uint32_t rawId() const { return mId_; }
  inline float noiseThreshold() const { return noiseThresh_; }
  inline float seedThreshold() const { return seedThresh_; }

  // Methods for HcalDbASCIIIO
  inline float getValue0() const { return noiseThresh_; }
  inline float getValue1() const { return seedThresh_; }

private:
  uint32_t mId_;
  float noiseThresh_;
  float seedThresh_;

  COND_SERIALIZABLE;
};

#endif  // CondFormats_HcalObjects_HcalPFCut_h
