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

HepGeom::Transform3D
CaloCellGeometry::getTransform( std::vector<HepGeom::Point3D<double> >* lptr ) const 
{
   const GlobalPoint& p ( CaloCellGeometry::getPosition() ) ;
   const HepGeom::Point3D<double>    gFront ( p.x(), p.y(), p.z() ) ;

   HepGeom::Point3D<double>  lFront ;
   assert(                               0 != param() ) ;
   std::vector<HepGeom::Point3D<double> > lc ( vocalCorners( param(), lFront ) ) ;

   HepGeom::Point3D<double>  lBack  ( 0.25*(lc[4]+lc[5]+lc[6]+lc[7]) ) ;

   const HepGeom::Point3D<double>  lOne  ( lc[0] ) ;

   const CornersVec& cor ( getCorners() ) ;
   std::vector<HepGeom::Point3D<double> > kor ( 8, HepGeom::Point3D<double> (0,0,0) ) ;
   for( unsigned int i ( 0 ) ; i != 8 ; ++i )
   {
      kor[i] = HepGeom::Point3D<double> ( cor[i].x(), cor[i].y(), cor[i].z() ) ;
   }

   HepGeom::Point3D<double>  gBack ( 0.25*( kor[4]+kor[5]+kor[6]+kor[7] ) ) ;

   const HepGeom::Vector3D<double>  gAxis ( (gBack-gFront).unit() ) ;

   gBack = ( gFront + (lBack-lFront).mag()*gAxis ) ;
   const HepGeom::Point3D<double>  gOneT ( gFront + ( lOne - lFront ).mag()*( kor[0] - gFront ).unit() ) ;

   const double langle ( ( lBack - lFront).angle( lOne - lFront ) ) ;
   const double gangle ( ( gBack - gFront).angle( gOneT- gFront ) ) ;
   const double dangle ( langle - gangle ) ;

   const HepGeom::Plane3D<double>  gPl (  gFront, gOneT, gBack ) ;
   const HepGeom::Point3D<double>  p2  ( gFront + gPl.normal().unit() ) ;

   const HepGeom::Point3D<double>  gOne ( gFront + HepGeom::Rotate3D( -dangle, gFront, p2 )*
			   HepGeom::Vector3D<double> ( gOneT - gFront ) ) ;

   const HepGeom::Transform3D tr ( lFront , lBack , lOne ,
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
