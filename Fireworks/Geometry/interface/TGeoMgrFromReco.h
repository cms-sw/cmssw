#ifndef GEOMETRY_TGEO_MGR_FROM_RECO_H
# define GEOMETRY_TGEO_MGR_FROM_RECO_H

# include <string>
# include <map>

# include "boost/shared_ptr.hpp"

# include "FWCore/Framework/interface/ESProducer.h"
# include "FWCore/Framework/interface/ESTransientHandle.h"
# include "DataFormats/GeometryVector/interface/GlobalPoint.h"
# include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
# include "Geometry/CSCGeometry/interface/CSCGeometry.h"
# include "Geometry/DTGeometry/interface/DTGeometry.h"
# include "Geometry/RPCGeometry/interface/RPCGeometry.h"
# include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

namespace edm
{
   class ParameterSet;
}

class DisplayTrackingGeomRecord;

class TGeoManager;
class TGeoShape;
class TGeoVolume;
class TGeoMaterial;
class TGeoMedium;
class GeomDet;

class TGeoMgrFromReco : public edm::ESProducer
{
public:
  TGeoMgrFromReco( const edm::ParameterSet& );
  virtual ~TGeoMgrFromReco( void );
  
  typedef boost::shared_ptr<TGeoManager> ReturnType;
  ReturnType produce( const DisplayTrackingGeomRecord& );

private:
  TGeoMgrFromReco( const TGeoMgrFromReco& );
  const TGeoMgrFromReco& operator=( const TGeoMgrFromReco& );

  TGeoManager*  createManager( int level );
  TGeoShape* createShape( const GeomDet *det );
  TGeoVolume* createVolume( const std::string& name, const GeomDet *det, const std::string& matname = "Air" );
  TGeoMaterial* createMaterial( const std::string& name );
  const std::string path( TGeoVolume* top, const std::string& name, int copy );

  void addCSCGeometry( TGeoVolume* top, const std::string& name = "CSC", int copy = 1 );
  void addDTGeometry( TGeoVolume* top, const std::string& name = "DT", int copy = 1 );
  void addRPCGeometry( TGeoVolume* top, const std::string& name = "RPC", int copy = 1 );
  void addPixelBarrelGeometry( TGeoVolume* top, const std::string& name = "PixelBarrel", int copy = 1 );
  void addPixelForwardGeometry( TGeoVolume* top, const std::string& name = "PixelForward", int copy = 1 );
  void addTIBGeometry( TGeoVolume* top, const std::string& name = "TIB", int copy = 1 );
  void addTOBGeometry( TGeoVolume* top, const std::string& name = "TOB", int copy = 1 );
  void addTIDGeometry( TGeoVolume* top, const std::string& name = "TID", int copy = 1 );
  void addTECGeometry( TGeoVolume* top, const std::string& name = "TEC", int copy = 1 );

  int                      m_level;
  bool                     m_verbose;
  struct Info
  {
    std::string name;
    float points[24]; // x1,y1,z1...x8,y8,z8
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
	for( unsigned int i = 0; i < 24; ++i ) points[i] = 0;
	for( unsigned int i = 0; i < 9; ++i ) topology[i] = 0;
      }
    void
    fillPoints( std::vector<GlobalPoint>::const_iterator begin, std::vector<GlobalPoint>::const_iterator end )
      {
	 unsigned int index( 0 );
	 for( std::vector<GlobalPoint>::const_iterator i = begin; i != end; ++i )
	 {
	    assert( index < 8 );
	    points[index*3] = i->x();
	    points[index*3+1] = i->y();
	    points[index*3+2] = i->z();
	    ++index;
	 }
      }
  };

  std::map<std::string, TGeoShape*>    m_nameToShape;
  std::map<std::string, TGeoVolume*>   m_nameToVolume;
  std::map<std::string, TGeoMaterial*> m_nameToMaterial;
  std::map<std::string, TGeoMedium*>   m_nameToMedium;
  std::map<unsigned int, Info>         m_idToName;

  edm::ESTransientHandle<GlobalTrackingGeometry> m_geomRecord;
  const TrackerGeometry* m_trackerGeom;
  const RPCGeometry*     m_rpcGeom;
};

#endif // GEOMETRY_TGEO_MGR_FROM_RECO_H
