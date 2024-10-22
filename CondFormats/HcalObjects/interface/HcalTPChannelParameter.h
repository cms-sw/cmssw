#ifndef CondFormatsHcalObjectsHcalTPChannelParameter_h
#define CondFormatsHcalObjectsHcalTPChannelParameter_h

#include "CondFormats/Serialization/interface/Serializable.h"
#include <cstdint>

class HcalTPChannelParameter {
public:
  /// get mask for channel validity and self trigger information
  uint32_t getMask() const { return mask_; }
  /// get FG bit information
  uint32_t getFGBitInfo() const { return fgBitInfo_; }
  /// get Detector ID
  uint32_t rawId() const { return id_; }
  int getauxi1() const { return auxi1_; }
  int getauxi2() const { return auxi2_; }

  // functions below are not supposed to be used by consumer applications

  HcalTPChannelParameter() : id_(0), mask_(0), fgBitInfo_(0), auxi1_(0), auxi2_(0) {}

  HcalTPChannelParameter(uint32_t fId, uint32_t mask, uint32_t bitInfo, int auxi1 = 0, int auxi2 = 0)
      : id_(fId), mask_(mask), fgBitInfo_(bitInfo), auxi1_(auxi1), auxi2_(auxi2) {}

private:
  uint32_t id_;
  uint32_t mask_;
  uint32_t fgBitInfo_;
  int auxi1_;
  int auxi2_;

  COND_SERIALIZABLE;
};

#endif
