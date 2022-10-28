#ifndef DDI_Shapeless_h
#define DDI_Shapeless_h

#include <iostream>
#include "DetectorDescription/Core/interface/Solid.h"

namespace DDI {

  class Shapeless : public Solid {
  public:
    Shapeless() : Solid(DDSolidShape::ddshapeless) {}
    double volume() const override { return 0; }
    void stream(std::ostream& os) const override { os << " shapeless"; }
  };
}  // namespace DDI
#endif
