#ifndef GEOMETRY_FWTGEO_RECO_GEOMETRY_H
# define GEOMETRY_FWTGEO_RECO_GEOMETRY_H

# include <cassert>
# include <map>
# include <string>
# include <vector>

# include "DataFormats/GeometryVector/interface/GlobalPoint.h"

class TGeoManager;

class FWTGeoRecoGeometry
{
public:
  FWTGeoRecoGeometry( void );
  virtual ~FWTGeoRecoGeometry( void );

  struct Info
  {
    std::string name;
    float points[36]; // x1,y1,z1...x8,y8,z8,...,x12,y12,z12
    float topology[9]; 
    Info( const std::string& iname )
      : name( iname )
      {
	init();
      }
    Info( void )
      {
	init();
      }
    void
    init( void )
      {
	for( unsigned int i = 0; i < 36; ++i ) points[i] = 0;
	for( unsigned int i = 0; i < 9; ++i ) topology[i] = 0;
      }
    void
    fillPoints( std::vector<GlobalPoint>::const_iterator begin, std::vector<GlobalPoint>::const_iterator end )
      {
	 unsigned int index( 0 );
	 for( std::vector<GlobalPoint>::const_iterator i = begin; i != end; ++i )
	 {
	    assert( index < 12 );
	    points[index*3] = i->x();
	    points[index*3+1] = i->y();
	    points[index*3+2] = i->z();
	    ++index;
	 }
      }
  };
  typedef std::map<unsigned int, FWTGeoRecoGeometry::Info> InfoMap;

  InfoMap idToName;
  
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

#endif // GEOMETRY_FWTGEO_RECO_GEOMETRY_H
