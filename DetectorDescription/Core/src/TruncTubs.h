#ifndef DDD_DDI_TRUNCTUBS_H
#define DDD_DDI_TRUNCTUBS_H

#include <iostream>
#include "Solid.h"

namespace DDI {

  class TruncTubs : public Solid
  {
  public:
    TruncTubs(double zHalf,
              double rIn, double rOut,
	      double startPhi,
	      double deltaPhi,
	      double cutAtStart,
	      double cutAtDelta,
	      bool cutInside);
    
    double volume() const { return -1; }
    
    void stream(std::ostream & os) const;
   };
}

#endif

