#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include <CLHEP/Geometry/Plane3D.h>

const float CaloCellGeometry::k_ScaleFromDDDtoGeant ( 0.1 ) ;

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

HepTransform3D
CaloCellGeometry::getTransform( std::vector<HepPoint3D>* lptr ) const 
{
   const GlobalPoint& p ( CaloCellGeometry::getPosition() ) ;
   const HepPoint3D   gFront ( p.x(), p.y(), p.z() ) ;

   HepPoint3D lFront ;
   assert(                               0 != param() ) ;
   std::vector<HepPoint3D> lc ( vocalCorners( param(), lFront ) ) ;

   HepPoint3D lBack  ( 0.25*(lc[4]+lc[5]+lc[6]+lc[7]) ) ;

   const HepPoint3D lOne  ( lc[0] ) ;

   const CornersVec& cor ( getCorners() ) ;
   std::vector<HepPoint3D> kor ( 8, HepPoint3D(0,0,0) ) ;
   for( unsigned int i ( 0 ) ; i != 8 ; ++i )
   {
      kor[i] = HepPoint3D( cor[i].x(), cor[i].y(), cor[i].z() ) ;
   }

   HepPoint3D gBack ( 0.25*( kor[4]+kor[5]+kor[6]+kor[7] ) ) ;

   const HepVector3D gAxis ( (gBack-gFront).unit() ) ;

   gBack = ( gFront + (lBack-lFront).mag()*gAxis ) ;
   const HepPoint3D gOneT ( gFront + ( lOne - lFront ).mag()*( kor[0] - gFront ).unit() ) ;

   const double langle ( ( lBack - lFront).angle( lOne - lFront ) ) ;
   const double gangle ( ( gBack - gFront).angle( gOneT- gFront ) ) ;
   const double dangle ( langle - gangle ) ;

   const HepPlane3D gPl (  gFront, gOneT, gBack ) ;
   const HepPoint3D p2  ( gFront + gPl.normal().unit() ) ;

   const HepPoint3D gOne ( gFront + HepRotate3D( -dangle, gFront, p2 )*
			   HepVector3D( gOneT - gFront ) ) ;

   const HepTransform3D tr ( lFront , lBack , lOne ,
			     gFront , gBack , gOne    ) ;

   if( 0 != lptr ) (*lptr) = lc ;

   return tr ;
}

const double* 
CaloCellGeometry::checkParmPtr(
   const std::vector<double>&   vv  ,
   CaloCellGeometry::ParVecVec& pvv  )
{
   const double* pP ( 0 ) ;

   for( unsigned int ii ( 0 ) ; ii != pvv.size() ; ++ii )
   {
      const ParVec& v ( pvv[ii] ) ;
      assert( v.size() == vv.size() ) ;

      bool same ( true ) ;
      for( unsigned int j ( 0 ) ; j != vv.size() ; ++j )
      {
	 same = same && ( vv[j] == v[j] ) ;
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

const double* 
CaloCellGeometry::getParmPtr(
   const std::vector<double>&   vv  ,
   CaloCellGeometry::ParMgr*    mgr ,
   CaloCellGeometry::ParVecVec& pvv  )
{
   const double* pP ( checkParmPtr( vv, pvv ) ) ;

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
