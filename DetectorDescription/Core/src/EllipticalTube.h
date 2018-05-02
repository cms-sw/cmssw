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
     : Solid(DDSolidShape::ddellipticaltube)
    { 
      p_.emplace_back(xSemiAxis);
      p_.emplace_back(ySemiAxis);
      p_.emplace_back(zHeight);
    }  
    ~EllipticalTube() override { }

    /// Not as flexible and possibly less accurate than G4 volume.
    double volume() const override ;
    void stream(std::ostream & os) const override;
  };

}
#endif // DDI_EllipticalTube_h
