#ifndef EcalTPGFineGrainConstEB_h
#define EcalTPGFineGrainConstEB_h

#include "CondFormats/Serialization/interface/Serializable.h"
#include <cstdint>

class EcalTPGFineGrainConstEB {
public:
  EcalTPGFineGrainConstEB();
  ~EcalTPGFineGrainConstEB();

  void getValues(uint32_t& ThresholdETLow,
                 uint32_t& ThresholdETHigh,
                 uint32_t& RatioLow,
                 uint32_t& RatioHigh,
                 uint32_t& LUT) const;
  void setValues(const uint32_t& ThresholdETLow,
                 const uint32_t& ThresholdETHigh,
                 const uint32_t& RatioLow,
                 const uint32_t& RatioHigh,
                 const uint32_t& LUT);

private:
  uint32_t ThresholdETLow_;
  uint32_t ThresholdETHigh_;
  uint32_t RatioLow_;
  uint32_t RatioHigh_;
  uint32_t LUT_;

  COND_SERIALIZABLE;
};

#endif
