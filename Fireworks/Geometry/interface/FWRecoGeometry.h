#ifndef GEOMETRY_FWRECO_GEOMETRY_H
# define GEOMETRY_FWRECO_GEOMETRY_H

# include "Fireworks/Core/interface/FWRecoGeom.h"
#include "TObjArray.h"
class FWRecoGeometry
{
public:
  FWRecoGeometry( )
    { idToName.reserve(260000); }
  
  virtual ~FWRecoGeometry( ) 
    {}

  FWRecoGeom::InfoMap idToName;
  TObjArray extraDet;
};

#endif // GEOMETRY_FWRECO_GEOMETRY_H
