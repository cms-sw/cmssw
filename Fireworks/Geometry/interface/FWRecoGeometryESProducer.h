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
class FWRecoGeometry;
class FWRecoGeometryRecord;
class GeomDet;

class FWRecoGeometryESProducer : public edm::ESProducer
{
public:
  FWRecoGeometryESProducer( const edm::ParameterSet& );
  virtual ~FWRecoGeometryESProducer( void );
  
  std::shared_ptr<FWRecoGeometry> produce( const FWRecoGeometryRecord& );

private:
  FWRecoGeometryESProducer( const FWRecoGeometryESProducer& );
  const FWRecoGeometryESProducer& operator=( const FWRecoGeometryESProducer& );
  
  void addCSCGeometry( void );
  void addDTGeometry( void );
  void addRPCGeometry( void );
  void addGEMGeometry( void );
  void addME0Geometry( void );
  void addPixelBarrelGeometry( void );
  void addPixelForwardGeometry( void );
  void addTIBGeometry( void );
  void addTOBGeometry( void );
  void addTIDGeometry( void );
  void addTECGeometry( void );
  void addCaloGeometry( void );
  
  unsigned int insert_id( unsigned int id );
  void fillPoints( unsigned int id, std::vector<GlobalPoint>::const_iterator begin, std::vector<GlobalPoint>::const_iterator end );
  void fillShapeAndPlacement( unsigned int id, const GeomDet *det );
  
  edm::ESHandle<GlobalTrackingGeometry>      m_geomRecord;
  edm::ESHandle<CaloGeometry>                m_caloGeom;
  std::vector<edm::ESHandle<HGCalGeometry> > m_hgcalGeoms;
  const TrackerGeometry*                     m_trackerGeom;
  std::shared_ptr<FWRecoGeometry>          m_fwGeometry;
  
  unsigned int m_current;
  bool m_tracker;
  bool m_muon;
  bool m_calo;
};

#endif // GEOMETRY_FWRECO_GEOMETRY_ES_PRODUCER_H
