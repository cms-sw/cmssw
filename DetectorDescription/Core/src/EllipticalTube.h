#ifndef DDI_EllipticalTube_h
#define DDI_EllipticalTube_h

#include <iosfwd>
#include "Solid.h"
#include <DataFormats/GeometryVector/interface/Pi.h>

namespace DDI {

  class EllipticalTube : public Solid
  {
  public:
    EllipticalTube(double xHalf, double yHalf, double zHalf)
     : Solid(ddellipticaltube)
    { 
      p_.push_back(xHalf);
      p_.push_back(yHalf);
      p_.push_back(zHalf);
    }  
    ~EllipticalTube() { }

    /// Not as flexible and possibly less accurate than G4 volume.
    double volume() const ;
    void stream(std::ostream & os) const;
  };

}
#endif // DDI_EllipticalTube_h
