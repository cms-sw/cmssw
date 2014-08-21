#ifndef GEOMETRY_FWTGEORECO_GEOMETRY_ES_PRODUCER_H
# define GEOMETRY_FWTGEORECO_GEOMETRY_ES_PRODUCER_H

# include "boost/shared_ptr.hpp"

# include "FWCore/Framework/interface/ESProducer.h"
# include "FWCore/Framework/interface/ESHandle.h"
# include "DataFormats/GeometryVector/interface/GlobalPoint.h"

namespace edm
{
   class ParameterSet;
}

class CaloGeometry;
class GlobalTrackingGeometry;
class TrackerGeometry;
class FWTGeoRecoGeometry;
class FWTGeoRecoGeometryRecord;

class TGeoManager;
class TGeoShape;
class TGeoVolume;
class TGeoMaterial;
class TGeoMedium;
class GeomDet;
class CaloCellGeometry;
class FWTGeoRecoGeometryESProducer : public edm::ESProducer
{
public:
   FWTGeoRecoGeometryESProducer( const edm::ParameterSet& );
   virtual ~FWTGeoRecoGeometryESProducer( void );
  
   boost::shared_ptr<FWTGeoRecoGeometry> produce( const FWTGeoRecoGeometryRecord& );

private:
   FWTGeoRecoGeometryESProducer( const FWTGeoRecoGeometryESProducer& );
   const FWTGeoRecoGeometryESProducer& operator=( const FWTGeoRecoGeometryESProducer& );

   TGeoManager*      createManager( int level );
   TGeoShape*        createShape( const GeomDet *det );
   TGeoVolume*       createVolume( const std::string& name, const GeomDet *det, const std::string& matname = "Air" );
   TGeoMaterial*     createMaterial( const std::string& name );

   TGeoVolume*  GetDaughter(TGeoVolume* mother, const char* prefix, int id);
   TGeoVolume*  GetTopHolder(const char* prefix);

   void addPixelBarrelGeometry();
   void addPixelForwardGeometry();
   void addTIBGeometry();
   void addTOBGeometry();
   void addTIDGeometry();
   void addTECGeometry();
   void addCSCGeometry();
   void addDTGeometry();
   void addRPCGeometry();
   void addGEMGeometry();
   void addEcalCaloGeometry();
   void addHcalCaloGeometryBarrel();
   void addHcalCaloGeometryEndcap();
  
   std::map<std::string, TGeoShape*>    m_nameToShape;
   std::map<TGeoShape*, TGeoVolume*>   m_shapeToVolume;
   std::map<std::string, TGeoMaterial*> m_nameToMaterial;
   std::map<std::string, TGeoMedium*>   m_nameToMedium;

   edm::ESHandle<GlobalTrackingGeometry> m_geomRecord;
   edm::ESHandle<CaloGeometry>           m_caloGeom;
   const TrackerGeometry* m_trackerGeom;
  
   boost::shared_ptr<FWTGeoRecoGeometry> m_fwGeometry;

   TGeoMedium* m_dummyMedium;
};

#endif // GEOMETRY_FWTGEORECO_GEOMETRY_ES_PRODUCER_H
