#include "PhysicsTools/SelectorUtils/interface/strbitset.h"

namespace pat {

  strbitset operator&(const strbitset& l, const strbitset& r) {
    strbitset ret = r;
    ret &= l;
    return ret;
  }

  strbitset operator|(const strbitset& l, const strbitset& r) {
    strbitset ret = r;
    ret |= l;
    return ret;
  }

  strbitset operator^(const strbitset& l, const strbitset& r){
    strbitset ret = r;
    ret ^= l;
    return ret;
  }

}
