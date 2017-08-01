#ifndef DDI_Orb_h
#define DDI_Orb_h

#include <DataFormats/GeometryVector/interface/Pi.h>
#include <iosfwd>
#include <vector>

#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "Solid.h"

namespace DDI {

  class Orb : public Solid
  {
  public:
    Orb(double rMax)
     : Solid(ddorb)
    { 
      p_.push_back(rMax);
    }  
    ~Orb() override { }
    
    double volume() const override { return (4.*Geom::pi()*p_[0]*p_[0]*p_[0])/3.; }
    void stream(std::ostream & os) const override;
  };

}
#endif // DDI_Orb_h
