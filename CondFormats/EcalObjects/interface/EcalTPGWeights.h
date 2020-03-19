#ifndef EcalTPGWeights_h
#define EcalTPGWeights_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <map>
#include <cstdint>

class EcalTPGWeights {
public:
  EcalTPGWeights();
  ~EcalTPGWeights();

  void getValues(uint32_t& w0, uint32_t& w1, uint32_t& w2, uint32_t& w3, uint32_t& w4) const;
  void setValues(const uint32_t& w0, const uint32_t& w1, const uint32_t& w2, const uint32_t& w3, const uint32_t& w4);

private:
  uint32_t w0_;
  uint32_t w1_;
  uint32_t w2_;
  uint32_t w3_;
  uint32_t w4_;

  COND_SERIALIZABLE;
};

#endif
