#ifndef DDI_Box_h
#define DDI_Box_h

#include <iosfwd>
#include <vector>

#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "Solid.h"


namespace DDI {

  class Box : public Solid
  {
  public:
    Box(double xHalf, double yHalf, double zHalf)
      : Solid(DDSolidShape::ddbox)
    { 
      p_.emplace_back(xHalf);
      p_.emplace_back(yHalf);
      p_.emplace_back(zHalf);
    }  
    ~Box() override { }
    
    double volume() const override { return 8.*p_[0]*p_[1]*p_[2]; }
    void stream(std::ostream & os) const override;
  };

}
#endif // DDI_Box_h
