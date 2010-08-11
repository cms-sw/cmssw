#ifndef CORE_FWRECO_GEOM_H
# define CORE_FWRECO_GEOM_H

# include <map>

class FWRecoGeom
{
public:
  FWRecoGeom( void ) {}
  
  virtual ~FWRecoGeom( void ) {}

  struct Info
  {
    std::string name;
    std::vector<float> points; // x1,y1,z1...x8,y8,z8
    std::vector<float> topology;
    Info( const std::string& iname )
      : name( iname ),
	points( 24, 0 ),
	topology( 9, 0 )
      {}
    Info( void )
      : points( 24, 0 ),
	topology( 9, 0 )
      {}
  };
  
  typedef std::map<unsigned int, FWRecoGeom::Info> InfoMap;
  typedef std::map<unsigned int, FWRecoGeom::Info>::const_iterator InfoMapItr;
};

#endif // CORE_FWRECO_GEOM_H
