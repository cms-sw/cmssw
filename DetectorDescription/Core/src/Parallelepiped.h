#ifndef DDI_Parallelepiped_h
#define DDI_Parallelepiped_h

#include <DataFormats/GeometryVector/interface/Pi.h>
#include <iosfwd>
#include <vector>

#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "Solid.h"

namespace DDI {

  class Parallelepiped : public Solid
  {
  public:
    Parallelepiped(double xHalf, double yHalf, double zHalf,
		   double alpha, double theta, double phi)
     : Solid(ddparallelepiped)
    { 
      p_.push_back(xHalf);
      p_.push_back(yHalf);
      p_.push_back(zHalf);
      p_.push_back(alpha);
      p_.push_back(theta);
      p_.push_back(phi);
    }  
    ~Parallelepiped() override { }

    /// Not as flexible and possibly less accurate than G4 volume.
    double volume() const override ;
    void stream(std::ostream & os) const override;
  };

}
#endif // DDI_Parallelepiped_h
