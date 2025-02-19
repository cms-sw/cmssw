#include "Geometry/CaloGeometry/interface/CaloGenericDetId.h"
#include "Geometry/HcalTowerAlgo/interface/CaloTowerGeometry.h"

typedef CaloCellGeometry::CCGFloat CCGFloat ;
typedef CaloCellGeometry::Pt3D     Pt3D     ;
typedef CaloCellGeometry::Pt3DVec  Pt3DVec  ;

CaloTowerGeometry::CaloTowerGeometry() :
   m_cellVec ( k_NumberOfCellsForCorners ) 
{
}
  

CaloTowerGeometry::~CaloTowerGeometry()
{
}


unsigned int
CaloTowerGeometry::alignmentTransformIndexLocal( const DetId& id )
{
   const CaloGenericDetId gid ( id ) ;

   assert( gid.isCaloTower() ) ;

   const CaloTowerDetId cid ( id ) ;

   const unsigned int iea ( cid.ietaAbs() ) ;

   const unsigned int ip ( ( cid.iphi() - 1 )/4 ) ;

   const int izoff ( ( cid.zside() + 1 )/2 ) ;

   const unsigned int offset ( izoff*3*18) ;

   return ( offset + ip + ( CaloTowerDetId::kEndIEta < iea ? 36 :
			    ( CaloTowerDetId::kBarIEta < iea ? 18 : 0 ) ) ) ;
}

unsigned int
CaloTowerGeometry::alignmentTransformIndexGlobal( const DetId& id )
{
   return (unsigned int) DetId::Calo - 1 ;
}

void
CaloTowerGeometry::localCorners( Pt3DVec&        lc  ,
				 const CCGFloat* pv  ,
				 unsigned int    i   ,
				 Pt3D&           ref  )
{
   IdealObliquePrism::localCorners( lc, pv, ref ) ;
}

void
CaloTowerGeometry::newCell( const GlobalPoint& f1 ,
			    const GlobalPoint& f2 ,
			    const GlobalPoint& f3 ,
			    const CCGFloat*    parm ,
			    const DetId&       detId   ) 
{
   const CaloGenericDetId cgid ( detId ) ;

   assert( cgid.isCaloTower() ) ;

   const unsigned int di ( cgid.denseIndex() ) ;

   m_cellVec[ di ] = IdealObliquePrism( f1, cornersMgr(), parm ) ;
   m_validIds.push_back( detId ) ;
}

const CaloCellGeometry* 
CaloTowerGeometry::cellGeomPtr( uint32_t index ) const
{
   const CaloCellGeometry* cell ( &m_cellVec[ index ] ) ;
   return  ( m_cellVec.size() < index ||
	     0 == cell->param() ? 0 : cell ) ;
}
