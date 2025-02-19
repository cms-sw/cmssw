#ifndef DDI_Polycone_h
#define DDI_Polycone_h

#include "Solid.h"

namespace DDI {
 
  class Polycone : public Solid
  {
  public:
    Polycone( double startPhi, double deltaPhi,
              const std::vector<double> & z,
              const std::vector<double> & rmin,
              const std::vector<double> & rmax); 
	      
    Polycone(  double startPhi, double deltaPhi,
               const std::vector<double> & z,
	       const std::vector<double> & r);
    
    double volume() const;
  	       
  };		  
}
#endif // DDI_Polycone_h
