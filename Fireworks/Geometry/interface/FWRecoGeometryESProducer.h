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
  ~FWRecoGeometryESProducer( void ) override;
  
  std::unique_ptr<FWRecoGeometry> produce( const FWRecoGeometryRecord& );

private:
  FWRecoGeometryESProducer( const FWRecoGeometryESProducer& ) = delete;
  const FWRecoGeometryESProducer& operator=( const FWRecoGeometryESProducer& ) = delete;
  
  void addCSCGeometry( FWRecoGeometry& );
  void addDTGeometry( FWRecoGeometry& );
  void addRPCGeometry( FWRecoGeometry& );
  void addGEMGeometry( FWRecoGeometry& );
  void addME0Geometry( FWRecoGeometry& );
  void addPixelBarrelGeometry( FWRecoGeometry& );
  void addPixelForwardGeometry( FWRecoGeometry& );
  void addTIBGeometry( FWRecoGeometry& );
  void addTOBGeometry( FWRecoGeometry& );
  void addTIDGeometry( FWRecoGeometry& );
  void addTECGeometry( FWRecoGeometry& );
  void addCaloGeometry( FWRecoGeometry& );

  void addFTLGeometry( FWRecoGeometry& );
  

   
  void ADD_PIXEL_TOPOLOGY( unsigned int rawid, const GeomDet* detUnit, FWRecoGeometry& );
   

  unsigned int insert_id( unsigned int id, FWRecoGeometry& );
  void fillPoints( unsigned int id,
                   std::vector<GlobalPoint>::const_iterator begin,
                   std::vector<GlobalPoint>::const_iterator end,
                   FWRecoGeometry& );
  void fillShapeAndPlacement( unsigned int id, const GeomDet *det, FWRecoGeometry& );
  void writeTrackerParametersXML(FWRecoGeometry&);
   
  edm::ESHandle<GlobalTrackingGeometry>      m_geomRecord;
  const CaloGeometry*                        m_caloGeom;
  edm::ESHandle<FastTimeGeometry>            m_ftlBarrelGeom,m_ftlEndcapGeom;
  std::vector<edm::ESHandle<HGCalGeometry> > m_hgcalGeoms;
  const TrackerGeometry*                     m_trackerGeom;
  
  unsigned int m_current;
  bool m_tracker;
  bool m_muon;
  bool m_calo;
  bool m_timing;
};

#endif // GEOMETRY_FWRECO_GEOMETRY_ES_PRODUCER_H
