#include "Geometry/CaloGeometry/interface/CaloGenericDetId.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/ForwardGeometry/interface/CastorGeometry.h"
#include "Geometry/ForwardGeometry/interface/IdealCastorTrapezoid.h"
#include "CastorGeometryData.h"
#include <algorithm>

typedef CaloCellGeometry::CCGFloat CCGFloat ;
typedef CaloCellGeometry::Pt3D     Pt3D     ;
typedef CaloCellGeometry::Pt3DVec  Pt3DVec  ;

CastorGeometry::CastorGeometry() :
   theTopology( new CastorTopology ), 
   lastReqDet_(DetId::Detector(0)),
   lastReqSubdet_(0),
   m_ownsTopology ( true ),
   m_cellVec ( k_NumberOfCellsForCorners )
{}

CastorGeometry::CastorGeometry( const CastorTopology* topology ) :
   theTopology(topology), 
   lastReqDet_(DetId::Detector(0)),
   lastReqSubdet_(0),
   m_ownsTopology ( false ),
   m_cellVec ( k_NumberOfCellsForCorners )
{}


CastorGeometry::~CastorGeometry() 
{
  if( m_ownsTopology ) delete theTopology ;
}

DetId 
CastorGeometry::getClosestCell(const GlobalPoint& r) const
{
   DetId returnId ( 0 ) ;
   const std::vector<DetId>& detIds ( getValidDetIds() ) ;
   for(auto detId : detIds)
   {
      const CaloCellGeometry* cell ( getGeometry( detId ) ) ;
      if( nullptr != cell &&
	  cell->inside( r ) )
      {
	 returnId = detId ;
	 break ;
      }
   }
   return returnId ;
}



unsigned int
CastorGeometry::alignmentTransformIndexLocal( const DetId& id )
{
   const CaloGenericDetId gid ( id ) ;

   assert( gid.isCastor() ) ;

   return 0 ;
}

unsigned int
CastorGeometry::alignmentTransformIndexGlobal( const DetId& /*id*/ )
{
   return (unsigned int)DetId::Calo - 1 ;
}

void
CastorGeometry::localCorners( Pt3DVec&        lc ,
			      const CCGFloat* pv ,
			      unsigned int  /*i*/,
			      Pt3D&           ref )
{
   IdealCastorTrapezoid::localCorners( lc, pv, ref ) ;
}

void
CastorGeometry::newCell( const GlobalPoint& f1 ,
			 const GlobalPoint& /*f2*/ ,
			 const GlobalPoint& /*f3*/ ,
			 const CCGFloat*    parm ,
			 const DetId&       detId   ) 
{
   const CaloGenericDetId cgid ( detId ) ;
   
   assert( cgid.isCastor() ) ;

   const unsigned int di ( cgid.denseIndex() ) ;

   m_cellVec[ di ] = IdealCastorTrapezoid( f1, cornersMgr(), parm ) ;
   addValidID( detId ) ;
}

const CaloCellGeometry* 
CastorGeometry::cellGeomPtr( uint32_t index ) const
{
   const CaloCellGeometry* cell ( &m_cellVec[ index ] ) ;
   return ( m_cellVec.size() < index ||
	    nullptr == cell->param() ? nullptr : cell ) ;
}
