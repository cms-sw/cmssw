#ifndef HGCAL_COMMON_DATA_SHASHLIK_GEOMETRY_H
# define HGCAL_COMMON_DATA_SHASHLIK_GEOMETRY_H

# include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
# include "Geometry/CaloGeometry/interface/IdealObliquePrism.h"
# include "Geometry/CaloTopology/interface/ShashlikTopology.h"
# include "Geometry/Records/interface/ShashlikGeometryRecord.h"
# include "DataFormats/EcalDetId/interface/EKDetId.h"

class ShashlikGeometry: public CaloSubdetectorGeometry 
{
public:
  typedef std::vector<IdealObliquePrism> EKCellVec;

  typedef CaloCellGeometry::CCGFloat CCGFloat;
  typedef CaloCellGeometry::Pt3D     Pt3D;
  typedef CaloCellGeometry::Pt3DVec  Pt3DVec;
  
  explicit ShashlikGeometry( const ShashlikTopology& );
  virtual ~ShashlikGeometry( void );

  static std::string producerTag() { return "Shashlik"; }
  
  virtual void newCell( const GlobalPoint& ,
			const GlobalPoint& ,
			const GlobalPoint& ,
			const CCGFloat* ,
			const DetId& );
  
  virtual const CaloCellGeometry* getGeometry( const DetId& id ) const {
    return cellGeomPtr( m_topology.detId2denseId( id ));
  }
  
protected:
 
  virtual const CaloCellGeometry* cellGeomPtr( unsigned int index ) const; 
  virtual unsigned int indexFor( const DetId& id ) const { return  m_topology.detId2denseId( id ); }

private:
  
  const ShashlikTopology& m_topology;
  // FIXME: for 71X - mutable edm::AtomicPtrCache<std::vector<DetId> > m_ekIds;
  mutable std::vector<DetId> m_ekIds;
  CaloSubdetectorGeometry::IVec m_dins;
  
  EKCellVec m_ekCellVec;
};

#endif // HGCAL_COMMON_DATA_SHASHLIK_GEOMETRY_H
