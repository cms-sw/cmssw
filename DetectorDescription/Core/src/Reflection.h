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
    double volume() const override;
    void stream(std::ostream &) const override;
    const DDSolid & solid() const { return s_; } 
  private:
    DDSolid s_;  
  };      
}
#endif
