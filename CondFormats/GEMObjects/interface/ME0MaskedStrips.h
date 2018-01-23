#ifndef CondFormats_GEMObjects_ME0MaskedStrips_h
#define CondFormats_GEMObjects_ME0MaskedStrips_h

#include "CondFormats/Serialization/interface/Serializable.h"
#include <vector>

class ME0MaskedStrips
{
 public:
  struct MaskItem {
    int rawId;
    int strip;
    COND_SERIALIZABLE;
  };

  ME0MaskedStrips(){}
  ~ME0MaskedStrips(){}

  std::vector<MaskItem> const & getMaskVec() const {return maskVec_;}

 private:
  std::vector<MaskItem> maskVec_;

  COND_SERIALIZABLE;
};

#endif
