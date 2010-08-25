#ifndef GEOMETRY_FWRECO_GEOMETRY_ES_PRODUCER_H
# define GEOMETRY_FWRECO_GEOMETRY_ES_PRODUCER_H

# include <string>
# include <map>

# include "boost/shared_ptr.hpp"

# include "FWCore/Framework/interface/ESProducer.h"
# include "FWCore/Framework/interface/ESTransientHandle.h"
# include "DataFormats/GeometryVector/interface/GlobalPoint.h"

namespace edm
{
   class ParameterSet;
}

class CaloGeometry;
class GlobalTrackingGeometry;
class TrackerGeometry;
class FWRecoGeometry;
class FWRecoGeometryRecord;

class TGeoManager;
class TGeoShape;
class TGeoVolume;
class TGeoMaterial;
class TGeoMedium;
class GeomDet;

class FWRecoGeometryESProducer : public edm::ESProducer
{
public:
  FWRecoGeometryESProducer( const edm::ParameterSet& );
  virtual ~FWRecoGeometryESProducer( void );
  
  boost::shared_ptr<FWRecoGeometry> produce( const FWRecoGeometryRecord& );

private:
  FWRecoGeometryESProducer( const FWRecoGeometryESProducer& );
  const FWRecoGeometryESProducer& operator=( const FWRecoGeometryESProducer& );

  TGeoManager*      createManager( int level );
  TGeoShape*        createShape( const GeomDet *det );
  TGeoVolume*       createVolume( const std::string& name, const GeomDet *det, const std::string& matname = "Air" );
  TGeoMaterial*     createMaterial( const std::string& name );
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
  void addCaloGeometry( void );
  
  std::map<std::string, TGeoShape*>    m_nameToShape;
  std::map<std::string, TGeoVolume*>   m_nameToVolume;
  std::map<std::string, TGeoMaterial*> m_nameToMaterial;
  std::map<std::string, TGeoMedium*>   m_nameToMedium;

  edm::ESTransientHandle<GlobalTrackingGeometry> m_geomRecord;
  edm::ESTransientHandle<CaloGeometry>           m_caloGeom;
  const TrackerGeometry* m_trackerGeom;
  
  boost::shared_ptr<FWRecoGeometry> m_fwGeometry;
  unsigned int insert_id( unsigned int rawid, const std::string name = "" );
  void fillPoints( unsigned int id, std::vector<GlobalPoint>::const_iterator begin, std::vector<GlobalPoint>::const_iterator end );
  
  unsigned int m_current;
};

#endif // GEOMETRY_FWRECO_GEOMETRY_ES_PRODUCER_H
