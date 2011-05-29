#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include <CLHEP/Geometry/Plane3D.h>

typedef CaloCellGeometry::CCGFloat CCGFloat ;
typedef CaloCellGeometry::Pt3D     Pt3D     ;
typedef CaloCellGeometry::Pt3DVec  Pt3DVec  ;
typedef CaloCellGeometry::Tr3D     Tr3D     ;

typedef HepGeom::Vector3D<CCGFloat> Vec3D   ;
typedef HepGeom::Plane3D<CCGFloat>  Plane3D ;

const float CaloCellGeometry::k_ScaleFromDDDtoGeant ( 0.1 ) ;

CaloCellGeometry::CaloCellGeometry() :
   m_refPoint ( 0., 0., 0. ),
   m_corners  (  ) ,
   m_parms    ( (CCGFloat*) 0 ) 
{
}

CaloCellGeometry::CaloCellGeometry( const CaloCellGeometry& cell )
{
   *this = cell ;
}

CaloCellGeometry&
CaloCellGeometry::operator=( const CaloCellGeometry& cell )
{
   if( this != &cell )
   {
      m_refPoint = cell.m_refPoint ;
      m_corners  = cell.m_corners  ;
      m_parms    = cell.m_parms    ;
   }
   return *this ;
}

CaloCellGeometry::~CaloCellGeometry()
{
}

const GlobalPoint& 
CaloCellGeometry::getPosition() const 
{
   return m_refPoint ; 
}

bool 
CaloCellGeometry::emptyCorners() const 
{
   return m_corners.empty() ;
}

const CCGFloat* 
CaloCellGeometry::param() const 
{
   return m_parms ;
}

CaloCellGeometry::CaloCellGeometry( CornersVec::const_reference gp ,
				    const CornersMgr*           mgr,
				    const CCGFloat*             par ) :
   m_refPoint ( gp  ),
   m_corners  ( mgr ),
   m_parms    ( par ) 
{
}

CaloCellGeometry::CaloCellGeometry( const CornersVec& cv,
				    const CCGFloat*   par ) : 
   m_refPoint ( 0.25*( cv[0].x() + cv[1].x() + cv[2].x() + cv[3].x() ),
		0.25*( cv[0].y() + cv[1].y() + cv[2].y() + cv[3].y() ),
		0.25*( cv[0].z() + cv[1].z() + cv[2].z() + cv[3].z() )  ), 
   m_corners  ( cv ),
   m_parms    ( par ) 
{
}

CaloCellGeometry::CornersVec& 
CaloCellGeometry::setCorners() const 
{
   return m_corners ; 
}

const CaloCellGeometry::CornersVec&
CaloCellGeometry::getCorners() const
{
   return m_corners ;
}

std::ostream& operator<<( std::ostream& s, const CaloCellGeometry& cell ) 
{
   s << ", Center: " <<  cell.getPosition() << std::endl;

   if( cell.emptyCorners() )
   {
      s << "Corners vector is empty." << std::endl ;
   }
   else
   {
      const CaloCellGeometry::CornersVec& corners ( cell.getCorners() ) ;
      for ( unsigned int i ( 0 ) ; i != corners.size() ; ++i ) 
      {
	 s << "Corner: " << corners[ i ] << std::endl;
      }
   }
   return s ;
}

void
CaloCellGeometry::getTransform( Tr3D& tr , Pt3DVec* lptr ) const 
{
   const GlobalPoint& p ( CaloCellGeometry::getPosition() ) ;
   const Pt3D gFront ( p.x(), p.y(), p.z() ) ;

   Pt3D lFront ;
   assert( 0 != param() ) ;
   Pt3DVec lc ( 8, Pt3D(0,0,0) ) ;
   vocalCorners( lc, param(), lFront ) ;

   Pt3D lBack ( 0.25*(lc[4]+lc[5]+lc[6]+lc[7]) ) ;

   Pt3D lOne ( lc[0] ) ;

   const CornersVec& cor ( getCorners() ) ;
   Pt3DVec kor ( 8, Pt3D(0,0,0) ) ;
   for( unsigned int i ( 0 ) ; i != 8 ; ++i )
   {
      kor[i] = Pt3D ( cor[i].x(), cor[i].y(), cor[i].z() ) ;
   }

   Pt3D  gBack ( 0.25*( kor[4]+kor[5]+kor[6]+kor[7] ) ) ;

   const Vec3D gAxis ( (gBack-gFront).unit() ) ;

   gBack = ( gFront + (lBack-lFront).mag()*gAxis ) ;
   const Pt3D  gOneT ( gFront + ( lOne - lFront ).mag()*( kor[0] - gFront ).unit() ) ;

   const float langle ( ( lBack - lFront).angle( lOne - lFront ) ) ;
   const float gangle ( ( gBack - gFront).angle( gOneT- gFront ) ) ;
   const float dangle ( langle - gangle ) ;

   const Plane3D gPl ( gFront, gOneT, gBack ) ;
   const Pt3D    p2  ( gFront + gPl.normal().unit() ) ;

   const Pt3D  gOne ( gFront + HepGeom::Rotate3D( -dangle, gFront, p2 )*
		      Vec3D ( gOneT - gFront ) ) ;

   tr = Tr3D( lFront , lBack , lOne ,
	      gFront , gBack , gOne    ) ;

   if( 0 != lptr ) (*lptr) = lc ;
}

const float* 
CaloCellGeometry::checkParmPtr(
   const std::vector<float>&   vv  ,
   CaloCellGeometry::ParVecVec& pvv  )
{
   const float* pP ( 0 ) ;

   for( unsigned int ii ( 0 ) ; ii != pvv.size() ; ++ii )
   {
      const ParVec& v ( pvv[ii] ) ;
      assert( v.size() == vv.size() ) ;

      bool same ( true ) ;
      for( unsigned int j ( 0 ) ; j != vv.size() ; ++j )
      {
	 same = same && ( fabs( vv[j] - v[j] )<1.e-6 ) ;
	 if( !same ) break ;
      }
      if( same )
      {
	 pP = &(*v.begin()) ;
	 break ;
      }
   }
   return pP ;
}

const float* 
CaloCellGeometry::getParmPtr(
   const std::vector<float>&   vv  ,
   CaloCellGeometry::ParMgr*    mgr ,
   CaloCellGeometry::ParVecVec& pvv  )
{
   const float* pP ( checkParmPtr( vv, pvv ) ) ;

   if( 0 == pP )
   {
      pvv.push_back( ParVec( mgr ) ) ;
      ParVec& back ( pvv.back() ) ;
      for( unsigned int i ( 0 ) ; i != vv.size() ; ++i )
      {
	 back[i] = vv[i] ;
      }
      pP = &(*back.begin()) ;
   }
   return pP ;
}

bool 
CaloCellGeometry::inside( const GlobalPoint& point ) const
{
   bool ans ( false ) ;
   const Pt3D  p ( point.x(), point.y(), point.z() ) ;
   const CornersVec& cog ( getCorners() ) ;
   Pt3D  co[8] ;
   for( unsigned int i ( 0 ) ; i != 8 ; ++i )
   {
      co[i] = Pt3D ( cog[i].x(), cog[i].y(), cog[i].z() ) ;
   }

   const Plane3D  AA ( co[0], co[1], co[2] ) ; // z<0
   const Plane3D  BB ( co[6], co[5], co[4] ) ; // z>0

   if( AA.distance(p)*BB.distance(p) >= 0 )
   {
     const Plane3D  CC ( co[0], co[4], co[5] ) ; // x<0
     const Plane3D  DD ( co[2], co[6], co[7] ) ; // x>0
     if( CC.distance(p)*DD.distance(p) >= 0 )
     {
       const Plane3D  EE ( co[3], co[7], co[4] ) ; // y<0
       const Plane3D  FF ( co[1], co[5], co[6] ) ; // y>0
       if( EE.distance(p)*FF.distance(p) >= 0 )
       {
	   ans = true ;
       }
      }
   }
   return ans ;
}
