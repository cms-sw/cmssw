#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include <algorithm>
#include <iostream>
//#include "assert.h"

//----------------------------------------------------------------------

HepGeom::Transform3D
TruncatedPyramid::getTransform( std::vector<HepGeom::Point3D<double> >* lptr ) const 
{
   const GlobalPoint& p ( CaloCellGeometry::getPosition() ) ;
   const HepGeom::Point3D<double>    gFront ( p.x(), p.y(), p.z() ) ;

   const double dz ( param()[0] ) ;

   HepGeom::Point3D<double>  lFront ;
   assert(                               0 != param() ) ;
   std::vector<HepGeom::Point3D<double> > lc ( 11.2 > dz ?
				localCorners( param(), lFront ) :
				localCornersSwap( param(), lFront )  ) ;

   // figure out if reflction volume or not

   HepGeom::Point3D<double>  lBack  ( 0.25*(lc[4]+lc[5]+lc[6]+lc[7]) ) ;

   assert( 0 != m_corOne ) ;

   const double disl ( ( lFront - lc[0] ).mag() ) ;
   const double disr ( ( lFront - lc[3] ).mag() ) ;
   const double disg ( ( gFront - (*m_corOne) ).mag() ) ;

   const double dell ( fabs( disg - disl ) ) ;
   const double delr ( fabs( disg - disr ) ) ;

   if( 11.2<dz &&
       delr < dell ) // reflection volume if true
   {
      lc = localCornersReflection( param(), lFront ) ;
      lBack  = 0.25*( lc[4] + lc[5] + lc[6] + lc[7] ) ;
   }

   const HepGeom::Point3D<double>  lOne  ( lc[0] ) ;

   const HepGeom::Vector3D<double>  gAxis ( axis().x(), axis().y(), axis().z() ) ;


   const HepGeom::Point3D<double>  gBack ( gFront + (lBack-lFront).mag()*gAxis ) ;
   const HepGeom::Point3D<double>  gOneT ( gFront + ( lOne - lFront ).mag()*( (*m_corOne) - gFront ).unit() ) ;

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

const CaloCellGeometry::CornersVec& 
TruncatedPyramid::getCorners() const 
{
   const CornersVec& co ( CaloCellGeometry::getCorners() ) ;
   if( co.uninitialized() ) 
   {
      CornersVec& corners ( setCorners() ) ;

      std::vector<HepGeom::Point3D<double> > lc ;

      const HepGeom::Transform3D tr ( getTransform( &lc ) ) ;

      for( unsigned int i ( 0 ) ; i != 8 ; ++i )
      {
	 const HepGeom::Point3D<double>  corn ( tr*lc[i] ) ;
	 corners[i] = GlobalPoint( corn.x(), corn.y(), corn.z() ) ;
      }

      delete m_corOne ; // no longer needed
      m_corOne = 0 ;
   }
   return co ;
}

namespace truncPyr
{
   HepGeom::Point3D<double>  refl( const HepGeom::Point3D<double> & p )
   {
      return HepGeom::Point3D<double> ( -p.x(), p.y(), p.z() ) ;
   }
}

std::vector<HepGeom::Point3D<double> >
TruncatedPyramid::localCornersReflection( const double* pv,
					  HepGeom::Point3D<double> &   ref )
{
   using namespace truncPyr ;

   std::vector<HepGeom::Point3D<double> > lc ( localCorners( pv, ref ) ) ;
   HepGeom::Point3D<double>  tmp ;
/*
   tmp   = lc[0] ;
   lc[0] = refl( lc[2] ) ;
   lc[2] = refl( tmp   ) ;
   tmp   = lc[1] ;
   lc[1] = refl( lc[3] ) ;
   lc[3] = refl( tmp   ) ;
   tmp   = lc[4] ;
   lc[4] = refl( lc[6] ) ;
   lc[6] = refl( tmp   ) ;
   tmp   = lc[5] ;
   lc[5] = refl( lc[7] ) ;
   lc[7] = refl( tmp   ) ;
*/
   lc[0] = refl( lc[0] ) ;
   lc[1] = refl( lc[1] ) ;
   lc[2] = refl( lc[2] ) ;
   lc[3] = refl( lc[3] ) ;
   lc[4] = refl( lc[4] ) ;
   lc[5] = refl( lc[5] ) ;
   lc[6] = refl( lc[6] ) ;
   lc[7] = refl( lc[7] ) ;


   ref   = 0.25*( lc[0] + lc[1] + lc[2] + lc[3] ) ;
   return lc ;
}

std::vector<HepGeom::Point3D<double> >
TruncatedPyramid::localCorners( const double* pv,
				HepGeom::Point3D<double> &   ref )
{
   assert( 0 != pv ) ;

   const double dz ( pv[0] ) ;
   const double th ( pv[1] ) ;
   const double ph ( pv[2] ) ;
   const double h1 ( pv[3] ) ;
   const double b1 ( pv[4] ) ;
   const double t1 ( pv[5] ) ;
   const double a1 ( pv[6] ) ;
   const double h2 ( pv[7] ) ;
   const double b2 ( pv[8] ) ;
   const double t2 ( pv[9] ) ;
   const double a2 ( pv[10]) ;
  
   const double ta1 ( tan( a1 ) ) ; // lower plane
   const double ta2 ( tan( a2 ) ) ; // upper plane

   const double tth   ( tan( th )       ) ;
   const double tthcp ( tth * cos( ph ) ) ;
   const double tthsp ( tth * sin( ph ) ) ;

   const unsigned int off ( h1<h2 ? 0 :  4 ) ;

   std::vector<HepGeom::Point3D<double> > lc ( 8, HepGeom::Point3D<double> (0,0,0) ) ;

   lc[0+off] = HepGeom::Point3D<double> ( -dz*tthcp - h1*ta1 - b1, -dz*tthsp - h1 , -dz ); // (-,-,-)
   lc[1+off] = HepGeom::Point3D<double> ( -dz*tthcp + h1*ta1 - t1, -dz*tthsp + h1 , -dz ); // (-,+,-)
   lc[2+off] = HepGeom::Point3D<double> ( -dz*tthcp + h1*ta1 + t1, -dz*tthsp + h1 , -dz ); // (+,+,-)
   lc[3+off] = HepGeom::Point3D<double> ( -dz*tthcp - h1*ta1 + b1, -dz*tthsp - h1 , -dz ); // (+,-,-)
   lc[4-off] = HepGeom::Point3D<double> (  dz*tthcp - h2*ta2 - b2,  dz*tthsp - h2 ,  dz ); // (-,-,+)
   lc[5-off] = HepGeom::Point3D<double> (  dz*tthcp + h2*ta2 - t2,  dz*tthsp + h2 ,  dz ); // (-,+,+)
   lc[6-off] = HepGeom::Point3D<double> (  dz*tthcp + h2*ta2 + t2,  dz*tthsp + h2 ,  dz ); // (+,+,+)
   lc[7-off] = HepGeom::Point3D<double> (  dz*tthcp - h2*ta2 + b2,  dz*tthsp - h2 ,  dz ); // (+,-,+)

   ref   = 0.25*( lc[0] + lc[1] + lc[2] + lc[3] ) ;

   return lc ;
}

std::vector<HepGeom::Point3D<double> >
TruncatedPyramid::localCornersSwap( const double* pv,
				    HepGeom::Point3D<double> &   ref )
{
   std::vector<HepGeom::Point3D<double> > lc ( localCorners( pv, ref ) ) ;

   HepGeom::Point3D<double>  tmp ;
   tmp   = lc[0] ;
   lc[0] = lc[3] ;
   lc[3] = tmp   ;
   tmp   = lc[1] ;
   lc[1] = lc[2] ;
   lc[2] = tmp   ;
   tmp   = lc[4] ;
   lc[4] = lc[7] ;
   lc[7] = tmp   ;
   tmp   = lc[5] ;
   lc[5] = lc[6] ;
   lc[6] = tmp   ;

   ref   = 0.25*( lc[0] + lc[1] + lc[2] + lc[3] ) ;

   return lc ;
}


// the following function is static and a helper for the endcap & barrel loader classes
// when initializing from DDD: fills corners vector from trap params plus transform

void 
TruncatedPyramid::createCorners( const std::vector<double>&    pv ,
				 const HepGeom::Transform3D&         tr ,
				 CaloCellGeometry::CornersVec& co   )
{
   assert( 11 == pv.size() ) ;

   // to get the ordering right for fast sim, we have to use their convention
   // which were based on the old static geometry. Some gymnastics required here.

   const double dz ( pv[0] ) ;
   const double h1 ( pv[3] ) ;
   const double h2 ( pv[7] ) ;
   std::vector<HepGeom::Point3D<double> > ko ( 8, HepGeom::Point3D<double> (0,0,0) ) ;

   // if reflection, different things for barrel and endcap
   static const HepGeom::Vector3D<double>  x ( 1, 0, 0 ) ;
   static const HepGeom::Vector3D<double>  y ( 0, 1, 0 ) ;
   static const HepGeom::Vector3D<double>  z ( 0, 0, 1 ) ;
   const bool refl ( ( ( tr*x ).cross( tr*y ) ).dot( tr*z ) < 0 ) ; // has reflection!


   HepGeom::Point3D<double>  tmp ;
   std::vector<HepGeom::Point3D<double> > to ( localCorners( &pv.front(), tmp ) ) ;

   for( unsigned int i ( 0 ) ; i != 8 ; ++i )
   {
      ko[i] = tr * to[i] ; // apply transformation
   }

   if( refl || 
       h1>h2  )
   {
      if( 11.2 < dz ) //barrel
      {
	 if( !refl )
	 {
	    to[0] = ko[3] ;
	    to[1] = ko[2] ;
	    to[2] = ko[1] ;
	    to[3] = ko[0] ;
	    to[4] = ko[7] ;
	    to[5] = ko[6] ;
	    to[6] = ko[5] ;      
	    to[7] = ko[4] ;      
	 }
	 else
	 {
	    to[0] = ko[0] ;
	    to[1] = ko[1] ;
	    to[2] = ko[2] ;
	    to[3] = ko[3] ;
	    to[4] = ko[4] ;
	    to[5] = ko[5] ;
	    to[6] = ko[6] ;      
	    to[7] = ko[7] ;      
	 }
      }
      else //endcap
      {
	 to[0] = ko[0] ;
	 to[1] = ko[3] ;
	 to[2] = ko[2] ;
	 to[3] = ko[1] ;
	 to[4] = ko[4] ;
	 to[5] = ko[7] ;
	 to[6] = ko[6] ;      
	 to[7] = ko[5] ;      
      }
      copy( to.begin(), to.end(), ko.begin() ) ; // faster than ko = to ? maybe.
   }
   for( unsigned int i ( 0 ) ; i != 8 ; ++i )
   {
      const HepGeom::Point3D<double> & p ( ko[i] ) ;
      co[ i ] = GlobalPoint( p.x(), p.y(), p.z() ) ;
   }
}
//----------------------------------------------------------------------

bool 
TruncatedPyramid::inside( const GlobalPoint& point ) const
{
   bool ans ( false ) ;
   const HepGeom::Point3D<double>  p ( point.x(), point.y(), point.z() ) ;
   const CornersVec& cog ( getCorners() ) ;
   HepGeom::Point3D<double>  co[8] ;
   for( unsigned int i ( 0 ) ; i != 8 ; ++i )
   {
      co[i] = HepGeom::Point3D<double> ( cog[i].x(), cog[i].y(), cog[i].z() ) ;
   }

   const HepGeom::Plane3D<double>  AA ( co[0], co[1], co[2] ) ; // z<0
   const HepGeom::Plane3D<double>  BB ( co[6], co[5], co[4] ) ; // z>0

   if( AA.distance(p)*BB.distance(p) >= 0 )
   {
     const HepGeom::Plane3D<double>  CC ( co[0], co[4], co[5] ) ; // x<0
     const HepGeom::Plane3D<double>  DD ( co[2], co[6], co[7] ) ; // x>0
     if( CC.distance(p)*DD.distance(p) >= 0 )
     {
       const HepGeom::Plane3D<double>  EE ( co[3], co[7], co[4] ) ; // y<0
       const HepGeom::Plane3D<double>  FF ( co[1], co[5], co[6] ) ; // y>0
       if( EE.distance(p)*FF.distance(p) >= 0 )
       {
	   ans = true ;
       }
      }
   }
   return ans ;
}
//----------------------------------------------------------------------

std::ostream& operator<<( std::ostream& s, const TruncatedPyramid& cell ) 
{
   s << "Center: " <<  ( (const CaloCellGeometry&) cell).getPosition() << std::endl;
   const float thetaaxis ( cell.getThetaAxis() ) ;
   const float phiaxis   ( cell.getPhiAxis()   ) ;
   s << "Axis: " <<  thetaaxis << " " << phiaxis << std::endl ;
   const CaloCellGeometry::CornersVec& corners ( cell.getCorners() ) ;
   for ( unsigned int i=0 ; i != corners.size() ; ++i ) 
   {
      s << "Corner: " << corners[i] << std::endl;
   }
  return s ;
}
  
