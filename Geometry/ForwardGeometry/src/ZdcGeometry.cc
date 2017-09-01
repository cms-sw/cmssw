#include "Geometry/CaloGeometry/interface/CaloGenericDetId.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/ForwardGeometry/interface/ZdcGeometry.h"
#include "Geometry/ForwardGeometry/interface/IdealZDCTrapezoid.h"
#include "ZdcHardcodeGeometryData.h"
#include <algorithm>

typedef CaloCellGeometry::CCGFloat CCGFloat ;
typedef CaloCellGeometry::Pt3D     Pt3D     ;
typedef CaloCellGeometry::Pt3DVec  Pt3DVec  ;

ZdcGeometry::ZdcGeometry() :
   theTopology( new ZdcTopology ),
   lastReqDet_(DetId::Detector(0)), 
   lastReqSubdet_(0),
   m_ownsTopology ( true ),
   m_cellVec ( k_NumberOfCellsForCorners )
{}

ZdcGeometry::ZdcGeometry( const ZdcTopology* topology) :
   theTopology(topology),
   lastReqDet_(DetId::Detector(0)), 
   lastReqSubdet_(0),
   m_ownsTopology ( false ),
   m_cellVec ( k_NumberOfCellsForCorners )
{}

ZdcGeometry::~ZdcGeometry() 
{
  if( m_ownsTopology ) delete theTopology ;
}
/*
DetId ZdcGeometry::getClosestCell(const GlobalPoint& r) const
{
   DetId returnId ( 0 ) ;
   const std::vector<DetId>& detIds ( getValidDetIds() ) ;
   for( std::vector<DetId>::const_iterator it ( detIds.begin() ) ;
	it != detIds.end(); ++it )
   {
      const CaloCellGeometry* cell ( getGeometry( *it ) ) ;
      if( 0 != cell &&
	  cell->inside( r ) )
      {
	 returnId = *it ;
	 break ;
      }
   }
   return returnId ;
}
*/
unsigned int
ZdcGeometry::alignmentTransformIndexLocal( const DetId& id )
{
   const CaloGenericDetId gid ( id ) ;

   assert( gid.isZDC() ) ;

   return ( 0 > HcalZDCDetId( id ).zside() ? 0 : 1 ) ;
}

unsigned int
ZdcGeometry::alignmentTransformIndexGlobal( const DetId& /*id*/ )
{
   return (unsigned int)DetId::Calo - 1 ;
}

void
ZdcGeometry::localCorners( Pt3DVec&        lc  ,
			   const CCGFloat* pv ,
			   unsigned int    /*i*/  ,
			   Pt3D&           ref  )
{
   IdealZDCTrapezoid::localCorners( lc, pv, ref ) ;
}

void
ZdcGeometry::newCell( const GlobalPoint& f1 ,
		      const GlobalPoint& /*f2*/ ,
		      const GlobalPoint& /*f3*/ ,
		      const CCGFloat*    parm ,
		      const DetId&       detId   ) 
{
   const CaloGenericDetId cgid ( detId ) ;

   assert( cgid.isZDC() ) ;

   const unsigned int di ( cgid.denseIndex() ) ;

   m_cellVec[ di ] = IdealZDCTrapezoid( f1, cornersMgr(), parm ) ;
   addValidID( detId ) ;
}

const CaloCellGeometry* 
ZdcGeometry::cellGeomPtr( uint32_t index ) const
{
   const CaloCellGeometry* cell ( &m_cellVec[ index ] ) ;
   return ( m_cellVec.size() < index ||
	    nullptr == cell->param() ? nullptr : cell ) ;
}

