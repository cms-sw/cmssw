#ifndef BOOSTTYPEOBJ_H
#define BOOSTTYPEOBJ_H
#include "CondFormats/Common/interface/Serializable.h"

#include <boost/cstdint.hpp>

class boostTypeObj {
public:
  boost::int8_t a;
  boost::int16_t b;
  boost::uint8_t aa;
  boost::uint16_t bb;

  COND_SERIALIZABLE;
};
#endif
