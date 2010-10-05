#include "PhysicsTools/SelectorUtils/interface/strbitset.h"

namespace pat {


 const std::string strbitset::dummy_ = std::string("");


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

  std::ostream & operator<<(std::ostream & out, const strbitset::index_type & r) {
    out << r.i_;
    return out;
  }

}
