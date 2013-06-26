#ifndef DDI_Box_h
#define DDI_Box_h

#include <iosfwd>
#include "Solid.h"


namespace DDI {

  class Box : public Solid
  {
  public:
    Box(double xHalf, double yHalf, double zHalf)
     : Solid(ddbox)
    { 
      p_.push_back(xHalf);
      p_.push_back(yHalf);
      p_.push_back(zHalf);
    }  
    ~Box() { }
    
    double volume() const { return 8.*p_[0]*p_[1]*p_[2]; }
    void stream(std::ostream & os) const;
  };

}
#endif // DDI_Box_h
