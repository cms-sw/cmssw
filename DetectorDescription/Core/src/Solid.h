#ifndef DDI_Solid_h
#define DDI_Solid_h

#include <iosfwd>
#include <vector>
#include "DetectorDescription/Core/interface/DDSolidShapes.h"

namespace DDI {
  
  class Solid
  {
  public:
  
    Solid() : shape_(dd_not_init) { }
    
    Solid(DDSolidShape shape) : shape_(shape) { }
    
    virtual ~Solid() { }
    
    const std::vector<double> & parameters() const { return p_; }
    
    virtual double volume() const { return 0; }
    
    DDSolidShape shape() const { return shape_; }
    
    virtual void stream(std::ostream &) const;
    
    void setParameters(std::vector<double> const & p) { p_ = p;}

  protected:
    DDSolidShape shape_;
    std::vector<double> p_; 
  };
}

#endif // DDI_Solid_h
