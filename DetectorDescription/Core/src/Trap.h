#ifndef DDI_Trap_h
#define DDI_Trap_h

#include <iostream>
#include "Solid.h"

namespace DDI {

  class Trap : public Solid
  {
  public:
    Trap(double pDz, 
         double pTheta,
         double pPhi,
         double pDy1, double pDx1,double pDx2,
         double pAlp1,
         double pDy2, double pDx3, double pDx4,
         double pAlp2);
    
    double volume() const;
    
    void stream(std::ostream &) const;
  };

}

#endif
