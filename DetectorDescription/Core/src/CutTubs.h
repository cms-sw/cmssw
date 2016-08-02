#ifndef DDD_DDI_CUTTUBS_H
#define DDD_DDI_CUTTUBS_H

#include <iostream>
#include "Solid.h"
#include <array>

namespace DDI {

  class CutTubs : public Solid
  {
  public:
    CutTubs(double zHalf,
	    double rIn, double rOut,
	    double startPhi,
	    double deltaPhi,
	    std::array<double, 3> pLowNorm,
	    std::array<double, 3> pHighNorm);
    
    double volume() const { return -1; }
    
    void stream(std::ostream & os) const;
   };
}

#endif
