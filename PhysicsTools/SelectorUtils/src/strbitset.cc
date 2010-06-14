#include "PhysicsTools/SelectorUtils/interface/strbitset.h"


pat::strbitset operator&(const pat::strbitset& l, const pat::strbitset& r) {
  pat::strbitset ret = r;
  ret &= l;
  return ret;
}

pat::strbitset operator|(const pat::strbitset& l, const pat::strbitset& r) {
  pat::strbitset ret = r;
  ret |= l;
  return ret;
}

pat::strbitset operator^(const pat::strbitset& l, const pat::strbitset& r){
  pat::strbitset ret = r;
  ret ^= l;
  return ret;
}

