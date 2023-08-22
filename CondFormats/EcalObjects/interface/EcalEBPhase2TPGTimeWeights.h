#ifndef EcalEBPhase2TPGTimeWeights_h
#define EcalEBPhase2TPGTimeWeights_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <map>
#include <cstdint>

class EcalEBPhase2TPGTimeWeights {
public:
  EcalEBPhase2TPGTimeWeights();
  ~EcalEBPhase2TPGTimeWeights();

  void getValues(uint32_t& w0,
                 uint32_t& w1,
                 uint32_t& w2,
                 uint32_t& w3,
                 uint32_t& w4,
                 uint32_t& w5,
                 uint32_t& w6,
                 uint32_t& w7,
                 uint32_t& w8,
                 uint32_t& w9,
                 uint32_t& w10,
                 uint32_t& w11) const;
  void setValues(const uint32_t& w0,
                 const uint32_t& w1,
                 const uint32_t& w2,
                 const uint32_t& w3,
                 const uint32_t& w4,
                 const uint32_t& w5,
                 const uint32_t& w6,
                 const uint32_t& w7,
                 const uint32_t& w8,
                 const uint32_t& w9,
                 const uint32_t& w10,
                 const uint32_t& w11);

private:
  uint32_t w0_;
  uint32_t w1_;
  uint32_t w2_;
  uint32_t w3_;
  uint32_t w4_;
  uint32_t w5_;
  uint32_t w6_;
  uint32_t w7_;
  uint32_t w8_;
  uint32_t w9_;
  uint32_t w10_;
  uint32_t w11_;

  COND_SERIALIZABLE;
};

#endif
