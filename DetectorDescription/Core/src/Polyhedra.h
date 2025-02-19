#ifndef DDI_Polyhedra_h
#define DDI_Polyhedra_h

#include "Solid.h"

namespace DDI {
 
  class Polyhedra : public Solid
  {
  public:
    Polyhedra( int sides, double startPhi, double deltaPhi,
              const std::vector<double> & z,
              const std::vector<double> & rmin,
              const std::vector<double> & rmax); 
	      
    Polyhedra(  int sides, double startPhi, double deltaPhi,
               const std::vector<double> & z,
	       const std::vector<double> & r);
    
    double volume() const;
  	       
  };		  
}
#endif // DDI_Polyhedra_h
