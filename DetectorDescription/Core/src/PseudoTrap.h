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
     : Solid(ddpseudotrap)
     {
       p_.push_back(x1);
       p_.push_back(x2);
       p_.push_back(y1);
       p_.push_back(y2);
       p_.push_back(z);
       p_.push_back(radius);
       p_.push_back(minusZ);
     }
    
    ~PseudoTrap() override{ }
    
    double volume() const override { return -1; }
    
    void stream(std::ostream & os) const override;
  };
   
}

#endif // DDI_PseudoTrap_h
