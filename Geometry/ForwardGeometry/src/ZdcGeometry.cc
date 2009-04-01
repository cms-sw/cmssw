#include "Geometry/CaloGeometry/interface/CaloGenericDetId.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/ForwardGeometry/interface/ZdcGeometry.h"
#include "Geometry/ForwardGeometry/interface/IdealZDCTrapezoid.h"
#include "ZdcHardcodeGeometryData.h"

ZdcGeometry::ZdcGeometry() :
   theTopology( new ZdcTopology ),
   lastReqDet_(DetId::Detector(0)), 
   lastReqSubdet_(0),
   m_ownsTopology ( true )
{
}

ZdcGeometry::ZdcGeometry( const ZdcTopology* topology) :
   theTopology(topology),
   lastReqDet_(DetId::Detector(0)), 
   lastReqSubdet_(0),
   m_ownsTopology ( false )
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

   unsigned int index ( 0 ) ;// to be implemented

   return index ;
}

unsigned int
ZdcGeometry::alignmentTransformIndexGlobal( const DetId& id )
{
   return (unsigned int)DetId::Calo ;
}

std::vector<HepPoint3D> 
ZdcGeometry::localCorners( const double* pv,
			   unsigned int  i,
			   HepPoint3D&   ref )
{
   return ( calogeom::IdealZDCTrapezoid::localCorners( pv, ref ) ) ;
}

CaloCellGeometry* 
ZdcGeometry::newCell( const GlobalPoint& f1 ,
		      const GlobalPoint& f2 ,
		      const GlobalPoint& f3 ,
		      CaloCellGeometry::CornersMgr* mgr,
		      const double*      parm ,
		      const DetId&       detId   ) 
{
   const CaloGenericDetId cgid ( detId ) ;

   assert( cgid.isZDC() ) ;

   return ( new calogeom::IdealZDCTrapezoid( f1, mgr, parm ) ) ;
}

