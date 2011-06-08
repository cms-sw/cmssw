#ifndef DDI_Solid_h
#define DDI_Solid_h

#include <iosfwd>
#include <vector>
// #include <utility>
// #include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"

// class DDSolid;

namespace DDI {
  
  class Solid
  {
  
    //   friend class DDSolid;
  public:
  
    Solid() : shape_(dd_not_init) { }
    
    Solid(DDSolidShape shape) : shape_(shape) { }
    
    virtual ~Solid() { }
    
    const std::vector<double> & parameters() const { return p_; }
    
    virtual double volume() const { return 0; }
    
    DDSolidShape shape() const { return shape_; }
    
    // bool boolean() const { IMPLEMENTATION MISSING }
    
    // DDSolid::Composites comp IMPLEMENTATION MISSING
    virtual void stream(std::ostream &) const;
    
    void setParameters(std::vector<double> const & p) { p_ = p;}

  protected:
    DDSolidShape shape_;
    std::vector<double> p_; 
  };
  
}

#endif // DDI_Solid_h
