#ifndef CondFormats_GEMObjects_GEMMaskedStrips_h
#define CondFormats_GEMObjects_GEMMaskedStrips_h

#include "CondFormats/Serialization/interface/Serializable.h"
#include <vector>
#include <iostream>

class GEMMaskedStrips {
public:
  struct MaskItem {
    int rawId;
    int strip;
    COND_SERIALIZABLE;
  };

  GEMMaskedStrips() {}
  ~GEMMaskedStrips() {}

  std::vector<MaskItem> const& getMaskVec() const { return maskVec_; }
  void fillMaskVec(MaskItem m) { maskVec_.push_back(m); }

private:
  std::vector<MaskItem> maskVec_;

  COND_SERIALIZABLE;
};
#endif
