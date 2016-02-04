#ifndef DDI_Orb_h
#define DDI_Orb_h

#include <iosfwd>
#include "Solid.h"
#include <DataFormats/GeometryVector/interface/Pi.h>

namespace DDI {

  class Orb : public Solid
  {
  public:
    Orb(double rMax)
     : Solid(ddorb)
    { 
      p_.push_back(rMax);
    }  
    ~Orb() { }
    
    double volume() const { return (4.*Geom::pi()*p_[0]*p_[0]*p_[0])/3.; }
    void stream(std::ostream & os) const;
  };

}
#endif // DDI_Orb_h
