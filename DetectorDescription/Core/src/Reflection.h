#ifndef DDI_Reflection_h
#define DDI_Reflection_h

#include <iostream>
#include "Solid.h"
#include "DetectorDescription/Core/interface/DDSolid.h"

namespace DDI {

  class Reflection : public Solid
  {
  public:
    Reflection(const DDSolid & s);
    double volume() const;
    void stream(std::ostream &) const;
    const DDSolid & solid() const { return s_; } 
  private:
    DDSolid s_;  
  };      
}
#endif
