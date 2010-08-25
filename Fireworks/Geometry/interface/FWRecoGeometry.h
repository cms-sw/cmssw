#ifndef GEOMETRY_FWRECO_GEOMETRY_H
# define GEOMETRY_FWRECO_GEOMETRY_H

# include <cassert>
# include <map>
# include <string>
# include <vector>

# include "Fireworks/Core/interface/FWRecoGeom.h"

class TGeoManager;

class FWRecoGeometry
{
public:
  FWRecoGeometry( void )
    : idToName( 260000 ),
      m_manager( 0 )
    {}
  
  virtual ~FWRecoGeometry( void ) 
    {}

  FWRecoGeom::InfoMap idToName;
  
  TGeoManager* manager( void ) const
    {
      return m_manager;
    }
  void manager( TGeoManager* geom )
    {
      m_manager = geom;
    }
  
private:
  TGeoManager* m_manager;
};

#endif // GEOMETRY_FWRECO_GEOMETRY_H
