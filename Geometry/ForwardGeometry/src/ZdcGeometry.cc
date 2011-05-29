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
{
}

ZdcGeometry::ZdcGeometry( const ZdcTopology* topology) :
   theTopology(topology),
   lastReqDet_(DetId::Detector(0)), 
   lastReqSubdet_(0),
   m_ownsTopology ( false ),
   m_cellVec ( k_NumberOfCellsForCorners )
{
}

ZdcGeometry::~ZdcGeometry() 
{
   if( m_ownsTopology ) delete theTopology ;
}

const std::vector<DetId>& 
ZdcGeometry::getValidDetIds( DetId::Detector det,
			     int             subdet ) const 
{
   const std::vector<DetId>& baseIds ( CaloSubdetectorGeometry::getValidDetIds() ) ;
   if( det    == DetId::Detector( 0 ) &&
       subdet == 0                        )
   {
      return baseIds ;
   }
   
   if( lastReqDet_    != det    ||
       lastReqSubdet_ != subdet    ) 
   {
      lastReqDet_     = det    ;
      lastReqSubdet_  = subdet ;
      m_validIds.clear();
      m_validIds.reserve( baseIds.size() ) ;
   }

   if( m_validIds.empty() ) 
   {
      for( unsigned int i ( 0 ) ; i != baseIds.size() ; ++i ) 
      {
	 const DetId id ( baseIds[i] );
	 if( id.det()      == det    &&
	     id.subdetId() == subdet    )
	 { 
	    m_validIds.push_back( id ) ;
	 }
      }
      std::sort(m_validIds.begin(),m_validIds.end());
   }
   return m_validIds;
}

DetId ZdcGeometry::getClosestCell(const GlobalPoint& r) const
{
   DetId returnId ( 0 ) ;
   const std::vector<DetId>& detIds ( getValidDetIds() ) ;
   for( std::vector<DetId>::const_iterator it ( detIds.begin() ) ;
	it != detIds.end(); ++it )
   {
      const CaloCellGeometry& cell ( *getGeometry( *it ) ) ;
      if( cell.inside( r ) )
      {
	 returnId = *it ;
	 break ;
      }
   }
   return returnId ;
}

unsigned int
ZdcGeometry::alignmentTransformIndexLocal( const DetId& id )
{
   const CaloGenericDetId gid ( id ) ;

   assert( gid.isZDC() ) ;

   return ( 0 > HcalZDCDetId( id ).zside() ? 0 : 1 ) ;
}

unsigned int
ZdcGeometry::alignmentTransformIndexGlobal( const DetId& id )
{
   return (unsigned int)DetId::Calo - 1 ;
}

void
ZdcGeometry::localCorners( Pt3DVec&        lc  ,
			   const CCGFloat* pv ,
			   unsigned int    i  ,
			   Pt3D&           ref  )
{
   IdealZDCTrapezoid::localCorners( lc, pv, ref ) ;
}

CaloCellGeometry* 
ZdcGeometry::newCell( const GlobalPoint& f1 ,
		      const GlobalPoint& f2 ,
		      const GlobalPoint& f3 ,
		      CaloCellGeometry::CornersMgr* mgr,
		      const CCGFloat*    parm ,
		      const DetId&       detId   ) 
{
   const CaloGenericDetId cgid ( detId ) ;

   assert( cgid.isZDC() ) ;

   const unsigned int di ( cgid.denseIndex() ) ;

   m_cellVec[ di ] = IdealZDCTrapezoid( f1, mgr, parm ) ;

   return &m_cellVec[ di ] ;
}

