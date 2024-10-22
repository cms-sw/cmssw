#ifndef RPCMaskedStrips_h
#define RPCMaskedStrips_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <iostream>

class RPCMaskedStrips {
public:
  struct MaskItem {
    int rawId;
    int strip;

    COND_SERIALIZABLE;
  };

  RPCMaskedStrips() {}

  ~RPCMaskedStrips() {}

  std::vector<MaskItem> const& getMaskVec() const { return MaskVec; }

  std::vector<MaskItem> MaskVec;

  COND_SERIALIZABLE;
};

#endif
