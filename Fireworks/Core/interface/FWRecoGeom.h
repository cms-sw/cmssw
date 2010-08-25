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
    unsigned int id;
    std::string name;
    float points[24]; // x1,y1,z1...x8,y8,z8
    float topology[9];

    bool operator< ( const Info& o ) const { 
      return ( this->id < o.id );
    }
  };
  
  typedef std::vector<FWRecoGeom::Info> InfoMap;
  typedef std::vector<FWRecoGeom::Info>::const_iterator InfoMapItr;
};

#endif // CORE_FWRECO_GEOM_H
