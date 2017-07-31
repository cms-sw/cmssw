#ifndef DDI_Ellipsoid_h
#define DDI_Ellipsoid_h

#include <DataFormats/GeometryVector/interface/Pi.h>
#include <iosfwd>
#include <vector>

#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "Solid.h"

namespace DDI {

  class Ellipsoid : public Solid
  {
  public:
    Ellipsoid(double  xSemiAxis,
	      double  ySemiAxis,
	      double  zSemiAxis,
	      double  zBottomCut=0,
	      double  zTopCut=0
	      )
      : Solid(ddellipsoid)
      { 
	p_.emplace_back(xSemiAxis);
	p_.emplace_back(ySemiAxis);
	p_.emplace_back(zSemiAxis);
	p_.emplace_back(zBottomCut);
	p_.emplace_back(zTopCut);
      }  
      ~Ellipsoid() override { }
      
      /// Not as flexible and possibly less accurate than G4 volume.
      double volume() const override ;
      double halfVol (double dz, double maxz) const;
      void stream(std::ostream & os) const override;
  };
  
}
#endif // DDI_Ellipsoid_h
