#ifndef GEOMETRY_FWRECO_GEOMETRY_ES_PRODUCER_H
# define GEOMETRY_FWRECO_GEOMETRY_ES_PRODUCER_H

# include <memory>

# include "FWCore/Framework/interface/ESProducer.h"
# include "FWCore/Framework/interface/ESHandle.h"
# include "DataFormats/GeometryVector/interface/GlobalPoint.h"

namespace edm
{
   class ParameterSet;
}

class CaloGeometry;
class HGCalGeometry;
class GlobalTrackingGeometry;
class TrackerGeometry;
class FastTimeGeometry;
class FWRecoGeometry;
class FWRecoGeometryRecord;
class GeomDet;

class FWRecoGeometryESProducer : public edm::ESProducer
{
public:
  FWRecoGeometryESProducer( const edm::ParameterSet& );
  virtual ~FWRecoGeometryESProducer( );
  
  std::shared_ptr<FWRecoGeometry> produce( const FWRecoGeometryRecord& );

private:
  FWRecoGeometryESProducer( const FWRecoGeometryESProducer& );
  const FWRecoGeometryESProducer& operator=( const FWRecoGeometryESProducer& );
  
  void addCSCGeometry( );
  void addDTGeometry( );
  void addRPCGeometry( );
  void addGEMGeometry( );
  void addME0Geometry( );
  void addPixelBarrelGeometry( );
  void addPixelForwardGeometry( );
  void addTIBGeometry( );
  void addTOBGeometry( );
  void addTIDGeometry( );
  void addTECGeometry( );
  void addCaloGeometry( );

  void addFTLGeometry( );
  

   
  void ADD_PIXEL_TOPOLOGY( unsigned int rawid, const GeomDet* detUnit );
   

  unsigned int insert_id( unsigned int id );
  void fillPoints( unsigned int id, std::vector<GlobalPoint>::const_iterator begin, std::vector<GlobalPoint>::const_iterator end );
  void fillShapeAndPlacement( unsigned int id, const GeomDet *det );
  
  edm::ESHandle<GlobalTrackingGeometry>      m_geomRecord;
  edm::ESHandle<CaloGeometry>                m_caloGeom;
  edm::ESHandle<FastTimeGeometry>            m_ftlBarrelGeom,m_ftlEndcapGeom;
  std::vector<edm::ESHandle<HGCalGeometry> > m_hgcalGeoms;
  const TrackerGeometry*                     m_trackerGeom;
  std::shared_ptr<FWRecoGeometry>            m_fwGeometry;
  
  unsigned int m_current;
  bool m_tracker;
  bool m_muon;
  bool m_calo;
  bool m_timing;
};

#endif // GEOMETRY_FWRECO_GEOMETRY_ES_PRODUCER_H
