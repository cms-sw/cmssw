#ifndef DDI_Torus_h
#define DDI_Torus_h

#include <iostream>
#include "Solid.h"

namespace DDI {

  class Torus : public Solid
  {
  public:
    Torus(double pRMin,
	  double pRMax,
	  double pRTor,
	  double pSPhi,
	  double pDPhi
	  );
    
    double volume() const;
    
    void stream(std::ostream &) const;
  };

}

#endif
