#ifndef DDI_Shapeless_h
#define DDI_Shapeless_h

#include <iostream>
#include "DetectorDescription/Core/src/Solid.h"

namespace DDI {

  class Shapeless : public Solid
  {
  public:
    Shapeless() : Solid(ddshapeless) { }
    double volume() const { return 0; }
    void stream(std::ostream & os) const 
     { os << " shapeless"; }
  };
}
#endif
