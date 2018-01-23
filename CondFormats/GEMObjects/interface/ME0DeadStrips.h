#ifndef CondFormats_ME0Objects_ME0DeadStrips_h
#define CondFormats_ME0Objects_ME0DeadStrips_h

#include "CondFormats/Serialization/interface/Serializable.h"
#include <vector>

class ME0DeadStrips
{
 public:
  struct DeadItem {
    int rawId;
    int strip;
    COND_SERIALIZABLE;
  };

  ME0DeadStrips(){}
  ~ME0DeadStrips(){}

  std::vector<DeadItem> const & getDeadVec() const {return deadVec_;}

 private:
  std::vector<DeadItem> deadVec_;

  COND_SERIALIZABLE;
};

#endif
