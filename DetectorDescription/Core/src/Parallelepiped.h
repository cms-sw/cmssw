#ifndef DDI_Parallelepiped_h
#define DDI_Parallelepiped_h

#include <iosfwd>
#include "Solid.h"
#include <DataFormats/GeometryVector/interface/Pi.h>

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
    ~Parallelepiped() { }

    /// Not as flexible and possibly less accurate than G4 volume.
    double volume() const ;
    void stream(std::ostream & os) const;
  };

}
#endif // DDI_Parallelepiped_h
