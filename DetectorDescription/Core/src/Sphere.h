#ifndef DDI_Sphere_h
#define DDI_Sphere_h

#include <iosfwd>
#include "Solid.h"

namespace DDI {
 
  class Sphere : public DDI::Solid
  {
  public:
    Sphere(double innerRadius,
	   double outerRadius,
	   double startPhi,
	   double deltaPhi,
	   double startZ,
	   double deltaZ);
   
    double volume() const ;
    void stream(std::ostream &) const;	 
  };

}

#endif
