#ifndef GEOMETRY_FWRECO_GEOMETRY_H
# define GEOMETRY_FWRECO_GEOMETRY_H

# include "Fireworks/Core/interface/FWRecoGeom.h"

class FWRecoGeometry
{
public:
  FWRecoGeometry( void )
    { idToName.reserve(260000); }
  
  virtual ~FWRecoGeometry( void ) 
    {}

  FWRecoGeom::InfoMap idToName;
};

#endif // GEOMETRY_FWRECO_GEOMETRY_H
