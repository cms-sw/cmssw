#ifndef DDI_Division_h
#define DDI_Division_h

#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDAxes.h"

#include <iostream>
#include <vector>
#include <utility>
#include <map>

namespace DDI {
  class Division {
    
  public:
    Division(const DDLogicalPart & parent,
	     const DDAxes axis,
	     const int nReplicas,
	     const double width,
	     const double offset );
      
      
    // Constructor with number of divisions 
    Division(const DDLogicalPart & parent,
	     const DDAxes axis,
	     const int nReplicas,
	     const double offset );
      
    // Constructor with width
    Division(const DDLogicalPart & parent,
	     const DDAxes axis,
	     const double width,
	     const double offset );
      
    DDAxes axis() const;
    int nReplicas() const;
    double width() const;
    double offset() const;
    const DDLogicalPart & parent() const;
    void stream(std::ostream &);
      
  private:
    DDLogicalPart parent_;
    DDAxes axis_;
    int nReplicas_;
    double width_;
    double offset_;
      
  };
}
#endif
