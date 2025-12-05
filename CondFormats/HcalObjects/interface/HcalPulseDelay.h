#ifndef CondFormats_HcalObjects_HcalPulseDelay_h
#define CondFormats_HcalObjects_HcalPulseDelay_h

#include <cstdint>
#include <string>

#include "CondFormats/Serialization/interface/Serializable.h"

class HcalPulseDelay {
public:
  inline HcalPulseDelay() : mId_(0), label_(), delay_(0.f) {}

  inline HcalPulseDelay(const unsigned long fId, const std::string& l,
                        const float t)
      : mId_(fId), label_(l), delay_(t) {}

  inline uint32_t rawId() const { return mId_; }
  inline const std::string& label() const { return label_; }
  inline float delay() const { return delay_;}

  // Methods for HcalDbASCIIIO
  inline const std::string& getValue0() const { return label_; }
  inline float getValue1() const { return delay_; }

private:
  uint32_t mId_;
  // Pulse label (tag)
  std::string label_;
  // Pulse delay in ns
  float delay_;

  COND_SERIALIZABLE;
};

#endif // CondFormats_HcalObjects_HcalPulseDelay_h
