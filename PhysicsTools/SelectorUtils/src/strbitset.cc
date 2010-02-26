#include "PhysicsTools/SelectorUtils/interface/strbitset.h"


std::strbitset operator&(const std::strbitset& l, const std::strbitset& r) {
  std::strbitset ret = r;
  ret &= l;
  return ret;
}

std::strbitset operator|(const std::strbitset& l, const std::strbitset& r) {
  std::strbitset ret = r;
  ret |= l;
  return ret;
}

std::strbitset operator^(const std::strbitset& l, const std::strbitset& r){
  std::strbitset ret = r;
  ret ^= l;
  return ret;
}

