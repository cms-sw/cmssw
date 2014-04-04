#include "Geometry/CaloGeometry/interface/CaloGenericDetId.h"
#include "Geometry/HcalTowerAlgo/interface/CaloTowerGeometry.h"

typedef CaloCellGeometry::CCGFloat CCGFloat ;
typedef CaloCellGeometry::Pt3D     Pt3D     ;
typedef CaloCellGeometry::Pt3DVec  Pt3DVec  ;

CaloTowerGeometry::CaloTowerGeometry(const CaloTowerTopology *cttopo_) :
  cttopo(cttopo_), 
  k_NumberOfCellsForCorners(cttopo->sizeForDenseIndexing()), 
  k_NumberOfShapes(cttopo->lastHFRing()), 
  m_cellVec ( k_NumberOfCellsForCorners ) 
{
}
  

CaloTowerGeometry::~CaloTowerGeometry() { }


unsigned int 
CaloTowerGeometry::alignmentTransformIndexLocal( const DetId& id ) {
  
  const CaloGenericDetId gid ( id ) ;
  assert( gid.isCaloTower() ) ;

  const CaloTowerDetId cid ( id ) ;
  const int iea ( cid.ietaAbs() ) ;
  const unsigned int ip ( ( cid.iphi() - 1 )/4 ) ;
  const int izoff ( ( cid.zside() + 1 )/2 ) ;
  const unsigned int offset ( izoff*3*18) ;
  
  assert(0);

  return ( offset + ip + 
	   ( cttopo->firstHFQuadPhiRing() <= iea ? 36 :
	     ( cttopo->firstHEDoublePhiRing() <= iea ? 18 : 0 ) ) ) ;
}

unsigned int
CaloTowerGeometry::alignmentTransformIndexGlobal( const DetId& id ) {
  return (unsigned int) DetId::Calo - 1 ;
}

void
CaloTowerGeometry::localCorners( Pt3DVec&        lc  ,
				 const CCGFloat* pv  ,
				 unsigned int    i   ,
				 Pt3D&           ref  ) {
  IdealObliquePrism::localCorners( lc, pv, ref ) ;
}

void
CaloTowerGeometry::newCell( const GlobalPoint& f1 ,
			    const GlobalPoint& f2 ,
			    const GlobalPoint& f3 ,
			    const CCGFloat*    parm ,
			    const DetId&       detId   ) {
  const CaloGenericDetId cgid ( detId ) ;

  assert( cgid.isCaloTower() ) ;
   
  const CaloTowerDetId cid ( detId ) ;

  const unsigned int di ( cttopo->denseIndex(cid) ) ;

  m_cellVec[ di ] = IdealObliquePrism( f1, cornersMgr(), parm ) ;
  m_validIds.push_back( detId ) ;
}

const CaloCellGeometry* 
CaloTowerGeometry::cellGeomPtr( uint32_t index ) const {
  const CaloCellGeometry* cell ( &m_cellVec[ index ] ) ;
  return  ( m_cellVec.size() < index ||
	    0 == cell->param() ? 0 : cell ) ;
}
