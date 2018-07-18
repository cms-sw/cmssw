#ifndef DDI_PseudoTrap_h
#define DDI_PseudoTrap_h

#include <iostream>
#include <vector>

#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "Solid.h"

namespace DDI {

  class PseudoTrap : public Solid
  {
  public:
    PseudoTrap(double x1, double x2, double y1, double y2, double z, double radius, bool minusZ)
     : Solid(DDSolidShape::ddpseudotrap)
     {
       p_.emplace_back(x1);
       p_.emplace_back(x2);
       p_.emplace_back(y1);
       p_.emplace_back(y2);
       p_.emplace_back(z);
       p_.emplace_back(radius);
       p_.emplace_back(minusZ);
     }
    
    ~PseudoTrap() override{ }
    
    double volume() const override { return -1; }
    
    void stream(std::ostream & os) const override;
  };
   
}

#endif // DDI_PseudoTrap_h
