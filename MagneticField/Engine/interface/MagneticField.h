#ifndef MagneticField_MagneticField_h
#define MagneticField_MagneticField_h

#include "Geometry/Vector/interface/GlobalVector.h"
#include "Geometry/Vector/interface/GlobalPoint.h"

class MagneticField
{
 public:
  MagneticField() {;}
  virtual ~MagneticField() {;}
  virtual GlobalVector inTesla ( const GlobalPoint& ) const = 0;
};

#endif
