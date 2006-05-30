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
  virtual GlobalVector inInverseGeV( const GlobalPoint& glb) const
    {return inTesla(glb)*2.99792458e-3;}
};

#endif
