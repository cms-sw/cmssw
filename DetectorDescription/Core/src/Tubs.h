#ifndef DDI_Tubs_h
#define DDI_Tubs_h

#include <iostream>
#include "Solid.h"

namespace DDI {

  class Tubs : public Solid
  {
  public:
    Tubs(double zhalf,
	 double rIn, double rOut,	      	      
	 double startPhi, 
	 double deltaPhi);

    double volume() const;
    
    void stream(std::ostream &) const;
  };  

}
#endif // DDI_Tubs_h
