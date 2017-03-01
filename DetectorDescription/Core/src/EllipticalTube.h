#ifndef DDI_EllipticalTube_h
#define DDI_EllipticalTube_h

#include <DataFormats/GeometryVector/interface/Pi.h>
#include <iosfwd>
#include <vector>

#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "Solid.h"

namespace DDI {

  class EllipticalTube : public Solid
  {
  public:
    EllipticalTube(double xSemiAxis, double ySemiAxis, double zHeight)
     : Solid(ddellipticaltube)
    { 
      p_.push_back(xSemiAxis);
      p_.push_back(ySemiAxis);
      p_.push_back(zHeight);
    }  
    ~EllipticalTube() { }

    /// Not as flexible and possibly less accurate than G4 volume.
    double volume() const ;
    void stream(std::ostream & os) const;
  };

}
#endif // DDI_EllipticalTube_h
