#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include <algorithm>
#include <iostream>


typedef TruncatedPyramid::CCGFloat CCGFloat ;
typedef TruncatedPyramid::Pt3D     Pt3D     ;
typedef TruncatedPyramid::Pt3DVec  Pt3DVec  ;
typedef TruncatedPyramid::Tr3D     Tr3D     ;

typedef HepGeom::Vector3D<CCGFloat> FVec3D   ;
typedef HepGeom::Plane3D<CCGFloat>  Plane3D ;

typedef HepGeom::Vector3D<double> DVec3D ;
typedef HepGeom::Plane3D<double>  DPlane3D ;
typedef HepGeom::Point3D<double>  DPt3D ;

//----------------------------------------------------------------------

TruncatedPyramid::TruncatedPyramid() : 
   CaloCellGeometry() ,
   m_axis ( 0., 0., 0. ),
   m_corOne ( 0., 0., 0. ) 
{}

TruncatedPyramid::TruncatedPyramid( const TruncatedPyramid& tr )
  : CaloCellGeometry( tr )
{
  *this = tr ; 
}

TruncatedPyramid& 
TruncatedPyramid::operator=( const TruncatedPyramid& tr ) 
{
   CaloCellGeometry::operator=( tr ) ;
   if( this != &tr )
   {
      m_axis   = tr.m_axis ;
      m_corOne = tr.m_corOne ; 
   }
   return *this ; 
}

TruncatedPyramid::TruncatedPyramid(       CornersMgr*  cMgr ,
				    const GlobalPoint& fCtr ,
				    const GlobalPoint& bCtr ,
				    const GlobalPoint& cor1 ,
				    const CCGFloat*    parV   ) :
   CaloCellGeometry ( fCtr, cMgr, parV ) ,
   m_axis           ( ( bCtr - fCtr ).unit() ) ,
   m_corOne         ( cor1.x(), cor1.y(), cor1.z() ) 
{initSpan();} 

TruncatedPyramid::TruncatedPyramid( const CornersVec& corn ,
				    const CCGFloat*   par    ) :
   CaloCellGeometry ( corn, par   ) , 
   m_axis           ( makeAxis()  ) ,
   m_corOne         ( corn[0].x(), corn[0].y(), corn[0].z()  )
{initSpan();} 

TruncatedPyramid::~TruncatedPyramid() 
{}

GlobalPoint TruncatedPyramid::getPosition( CCGFloat depth ) const
{
  return CaloCellGeometry::getPosition() + depth*m_axis ;
}

CCGFloat 
TruncatedPyramid::getThetaAxis() const 
{ 
   return m_axis.theta() ; 
} 

CCGFloat 
TruncatedPyramid::getPhiAxis() const 
{
   return m_axis.phi() ; 
} 

const GlobalVector& 
TruncatedPyramid::axis() const
{ 
   return m_axis ; 
}

void
TruncatedPyramid::vocalCorners( Pt3DVec&        vec ,
				const CCGFloat* pv  ,
				Pt3D&           ref  ) const
{ 
   localCorners( vec, pv, ref ) ; 
}

GlobalVector 
TruncatedPyramid::makeAxis() 
{ 
   return GlobalVector( backCtr() -
			CaloCellGeometry::getPosition() ).unit() ;
}

const GlobalPoint 
TruncatedPyramid::backCtr() const 
{
   return GlobalPoint( 0.25*( getCorners()[4].x() + getCorners()[5].x() +
			      getCorners()[6].x() + getCorners()[7].x() ),
		       0.25*( getCorners()[4].y() + getCorners()[5].y() +
			      getCorners()[6].y() + getCorners()[7].y() ),
		       0.25*( getCorners()[4].z() + getCorners()[5].z() +
			      getCorners()[6].z() + getCorners()[7].z() ) ) ;
}

void
TruncatedPyramid::getTransform( Tr3D& tr, Pt3DVec* lptr ) const 
{
   const GlobalPoint& p ( CaloCellGeometry::getPosition() ) ;
   const Pt3D         gFront ( p.x(), p.y(), p.z() ) ;
   const DPt3D        dgFront ( p.x(), p.y(), p.z() ) ;

   const double dz ( param()[0] ) ;

   Pt3D  lFront ;
   assert( nullptr != param() ) ;
   std::vector<Pt3D > lc( 8, Pt3D(0,0,0) ) ;
   if( 11.2 > dz )
   {
      localCorners( lc, param(), lFront ) ;
   }
   else
   {
      localCornersSwap( lc, param(), lFront ) ;
   }

   // figure out if reflction volume or not

   Pt3D  lBack  ( 0.25*(lc[4]+lc[5]+lc[6]+lc[7]) ) ;

   const double disl ( ( lFront - lc[0] ).mag() ) ;
   const double disr ( ( lFront - lc[3] ).mag() ) ;
   const double disg ( ( gFront - m_corOne ).mag() ) ;

   const double dell ( fabs( disg - disl ) ) ;
   const double delr ( fabs( disg - disr ) ) ;

   if( 11.2<dz &&
       delr < dell ) // reflection volume if true
   {
      localCornersReflection( lc, param(), lFront ) ;
      lBack  = 0.25*( lc[4] + lc[5] + lc[6] + lc[7] ) ;
   }

   const DPt3D dlFront ( lFront.x(), lFront.y(), lFront.z() ) ;
   const DPt3D dlBack  ( lBack.x() , lBack.y() , lBack.z()  ) ;
   const DPt3D dlOne   ( lc[0].x() , lc[0].y() , lc[0].z()  ) ;

   const FVec3D dgAxis  ( axis().x(), axis().y(), axis().z() ) ;

   const DPt3D dmOne   ( m_corOne.x(), m_corOne.y(), m_corOne.z() ) ;

   const DPt3D dgBack  ( dgFront + ( dlBack - dlFront ).mag()*dgAxis ) ;
   DPt3D dgOne ( dgFront + ( dlOne - dlFront ).mag()*( dmOne - dgFront ).unit() ) ;

   const double dlangle ( ( dlBack - dlFront).angle( dlOne - dlFront ) ) ;
   const double dgangle ( ( dgBack - dgFront).angle( dgOne - dgFront ) ) ;
   const double dangle  ( dlangle - dgangle ) ;

   if( 1.e-6 < fabs(dangle) )//guard against precision problems
   {
      const DPlane3D dgPl ( dgFront, dgOne, dgBack ) ;
      const DPt3D    dp2  ( dgFront + dgPl.normal().unit() ) ;

      DPt3D dgOld ( dgOne ) ;

      dgOne = ( dgFront + HepGeom::Rotate3D( -dangle, dgFront, dp2 )*
		DVec3D( dgOld - dgFront ) ) ;
   }

   tr = Tr3D( dlFront , dlBack , dlOne ,
	      dgFront , dgBack , dgOne    ) ;

   if( nullptr != lptr ) (*lptr) = lc ;
}

void
TruncatedPyramid::initCorners(CaloCellGeometry::CornersVec& corners) 
{
   if( corners.uninitialized() ) 
   {
      Pt3DVec lc ;

      Tr3D tr ;
      getTransform( tr, &lc ) ;

      for( unsigned int i ( 0 ) ; i != 8 ; ++i )
      {
	 const Pt3D corn ( tr*lc[i] ) ;
	 corners[i] = GlobalPoint( corn.x(), corn.y(), corn.z() ) ;
      }
   }
}

namespace truncPyr
{
   Pt3D  refl( const Pt3D & p )
   {
      return Pt3D ( -p.x(), p.y(), p.z() ) ;
   }
}

void
TruncatedPyramid::localCornersReflection( Pt3DVec&        lc  ,
					  const CCGFloat* pv  ,
		 			  Pt3D&           ref  )
{
//   using namespace truncPyr ;
   localCorners( lc, pv, ref ) ;
   Pt3D    tmp ;
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
   lc[0] = truncPyr::refl( lc[0] ) ;
   lc[1] = truncPyr::refl( lc[1] ) ;
   lc[2] = truncPyr::refl( lc[2] ) ;
   lc[3] = truncPyr::refl( lc[3] ) ;
   lc[4] = truncPyr::refl( lc[4] ) ;
   lc[5] = truncPyr::refl( lc[5] ) ;
   lc[6] = truncPyr::refl( lc[6] ) ;
   lc[7] = truncPyr::refl( lc[7] ) ;

   ref   = 0.25*( lc[0] + lc[1] + lc[2] + lc[3] ) ;
}

void
TruncatedPyramid::localCorners( Pt3DVec&        lc  ,
				const CCGFloat* pv  ,
				Pt3D&           ref   )
{
   assert( nullptr != pv ) ;
   assert( 8 == lc.size() ) ;

   const CCGFloat dz ( pv[TruncatedPyramid::k_Dz]    ) ;
   const CCGFloat th ( pv[TruncatedPyramid::k_Theta] ) ;
   const CCGFloat ph ( pv[TruncatedPyramid::k_Phi]   ) ;
   const CCGFloat h1 ( pv[TruncatedPyramid::k_Dy1]   ) ;
   const CCGFloat b1 ( pv[TruncatedPyramid::k_Dx1]   ) ;
   const CCGFloat t1 ( pv[TruncatedPyramid::k_Dx2]   ) ;
   const CCGFloat a1 ( pv[TruncatedPyramid::k_Alp1]  ) ;
   const CCGFloat h2 ( pv[TruncatedPyramid::k_Dy2]   ) ;
   const CCGFloat b2 ( pv[TruncatedPyramid::k_Dx3]   ) ;
   const CCGFloat t2 ( pv[TruncatedPyramid::k_Dx4]   ) ;
   const CCGFloat a2 ( pv[TruncatedPyramid::k_Alp2]  ) ;
  
   const CCGFloat ta1 ( tan( a1 ) ) ; // lower plane
   const CCGFloat ta2 ( tan( a2 ) ) ; // upper plane

   const CCGFloat tth   ( tan( th )       ) ;
   const CCGFloat tthcp ( tth * cos( ph ) ) ;
   const CCGFloat tthsp ( tth * sin( ph ) ) ;

   const unsigned int off ( h1<h2 ? 0 :  4 ) ;

   lc[0+off] = Pt3D ( -dz*tthcp - h1*ta1 - b1, -dz*tthsp - h1 , -dz ); // (-,-,-)
   lc[1+off] = Pt3D ( -dz*tthcp + h1*ta1 - t1, -dz*tthsp + h1 , -dz ); // (-,+,-)
   lc[2+off] = Pt3D ( -dz*tthcp + h1*ta1 + t1, -dz*tthsp + h1 , -dz ); // (+,+,-)
   lc[3+off] = Pt3D ( -dz*tthcp - h1*ta1 + b1, -dz*tthsp - h1 , -dz ); // (+,-,-)
   lc[4-off] = Pt3D (  dz*tthcp - h2*ta2 - b2,  dz*tthsp - h2 ,  dz ); // (-,-,+)
   lc[5-off] = Pt3D (  dz*tthcp + h2*ta2 - t2,  dz*tthsp + h2 ,  dz ); // (-,+,+)
   lc[6-off] = Pt3D (  dz*tthcp + h2*ta2 + t2,  dz*tthsp + h2 ,  dz ); // (+,+,+)
   lc[7-off] = Pt3D (  dz*tthcp - h2*ta2 + b2,  dz*tthsp - h2 ,  dz ); // (+,-,+)

   ref   = 0.25*( lc[0] + lc[1] + lc[2] + lc[3] ) ;
}

void
TruncatedPyramid::localCornersSwap( Pt3DVec&        lc  ,
				    const CCGFloat* pv  ,
				    Pt3D&           ref  )
{
   localCorners( lc, pv, ref ) ;

   Pt3D  tmp ;
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
}


// the following function is static and a helper for the endcap & barrel loader classes
// when initializing from DDD: fills corners vector from trap params plus transform

void 
TruncatedPyramid::createCorners( const std::vector<CCGFloat>&  pv ,
				 const Tr3D&                   tr ,
				 std::vector<GlobalPoint>&     co   )
{
   assert( 11 == pv.size() ) ;
   assert( 8 == co.size() ) ;
   // to get the ordering right for fast sim, we have to use their convention
   // which were based on the old static geometry. Some gymnastics required here.

   const CCGFloat dz ( pv[0] ) ;
   const CCGFloat h1 ( pv[3] ) ;
   const CCGFloat h2 ( pv[7] ) ;
   Pt3DVec        ko ( 8, Pt3D(0,0,0) ) ;

   // if reflection, different things for barrel and endcap
   static const FVec3D  x ( 1, 0, 0 ) ;
   static const FVec3D  y ( 0, 1, 0 ) ;
   static const FVec3D  z ( 0, 0, 1 ) ;
   const bool refl ( ( ( tr*x ).cross( tr*y ) ).dot( tr*z ) < 0 ) ; // has reflection!

   Pt3D    tmp ;
   Pt3DVec to ( 8, Pt3D(0,0,0) ) ;
   localCorners( to, &pv.front(), tmp ) ;

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
      const Pt3D & p ( ko[i] ) ;
      co[ i ] = GlobalPoint( p.x(), p.y(), p.z() ) ;
   }
}
//----------------------------------------------------------------------
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
  
