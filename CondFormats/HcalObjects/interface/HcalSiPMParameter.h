#ifndef CondFormatsHcalObjectsHcalSiPMParameter_h
#define CondFormatsHcalObjectsHcalSiPMParameter_h

#include "CondFormats/Serialization/interface/Serializable.h"
#include <cstdint>

class HcalSiPMParameter {
public:
  /// get SiPM type
  int getType() const { return type_; }
  /// get fcByPE
  float getFCByPE() const { return fcByPE_; }
  /// get dark current
  float getDarkCurrent() const { return darkCurrent_; }

  // functions below are not supposed to be used by consumer applications

  HcalSiPMParameter() : id_(0), type_(0), fcByPE_(0), darkCurrent_(0), auxi1_(0), auxi2_(0) {}

  HcalSiPMParameter(unsigned long fId, int type, float fcByPE, float darkCurrent, int auxi1 = 0, float auxi2 = 0)
      : id_(fId), type_(type), fcByPE_(fcByPE), darkCurrent_(darkCurrent), auxi1_(auxi1), auxi2_(auxi2) {}

  uint32_t rawId() const { return id_; }
  int getauxi1() const { return auxi1_; }
  float getauxi2() const { return auxi2_; }

private:
  uint32_t id_;
  int type_;
  float fcByPE_;
  float darkCurrent_;
  int auxi1_;
  float auxi2_;

  COND_SERIALIZABLE;
};

#endif
