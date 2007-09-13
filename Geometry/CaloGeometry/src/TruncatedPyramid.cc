#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include <algorithm>
#include <iostream>
//#include "assert.h"

//----------------------------------------------------------------------

// the following function is static and a helper for the endcap & barrel loader classes
// when initializing from DDD: fills corners vector from trap params plus transform

void 
TruncatedPyramid::createCorners( const std::vector<double>&    pv ,
				 const HepTransform3D&         tr ,
				 CaloCellGeometry::CornersVec& co   )
{
   assert( 11 == pv.size() ) ;

   // to get the ordering right for fast sim, we have to use their convention
   // which were based on the old static geometry. Some gymnastics required here.

   static const float kf ( CaloCellGeometry::k_ScaleFromDDDtoGeant ) ;

   const double dz ( kf*pv[0] ) ;
   const double th ( kf*pv[1] ) ;
   const double ph ( kf*pv[2] ) ;
   const double h1 ( kf*pv[3] ) ;
   const double b1 ( kf*pv[4] ) ;
   const double t1 ( kf*pv[5] ) ;
   const double a1 ( kf*pv[6] ) ;
   const double h2 ( kf*pv[7] ) ;
   const double b2 ( kf*pv[8] ) ;
   const double t2 ( kf*pv[9] ) ;
   const double a2 ( kf*pv[10]) ;

   std::vector<HepPoint3D> to ( 8, HepPoint3D(0,0,0) ) ;
   std::vector<HepPoint3D> ko ( 8, HepPoint3D(0,0,0) ) ;
  
   const double ta1 ( tan( a1 ) ) ; // lower plane
   const double ta2 ( tan( a2 ) ) ; // upper plane

   const double tth   ( tan( th )       ) ;
   const double tthcp ( tth * cos( ph ) ) ;
   const double tthsp ( tth * sin( ph ) ) ;

   const unsigned int off ( h1<h2 ? 0 :  4 ) ;

   to[0+off] = HepPoint3D( -dz*tthcp - h1*ta1 - b1, -dz*tthsp - h1 , -dz ); // (-,-,-)
   to[1+off] = HepPoint3D( -dz*tthcp + h1*ta1 - t1, -dz*tthsp + h1 , -dz ); // (-,+,-)
   to[2+off] = HepPoint3D( -dz*tthcp + h1*ta1 + t1, to[1].y()      , -dz ); // (+,+,-)
   to[3+off] = HepPoint3D( -dz*tthcp - h1*ta1 + b1, to[0].y()      , -dz ); // (+,-,-)
   to[4-off] = HepPoint3D(  dz*tthcp - h2*ta2 - b2, -dz*tthsp - h2 ,  dz ); // (-,-,+)
   to[5-off] = HepPoint3D(  dz*tthcp + h2*ta2 - t2, -dz*tthsp + h2 ,  dz ); // (-,+,+)
   to[6-off] = HepPoint3D(  dz*tthcp + h2*ta2 + t2, to[5].y()      ,  dz ); // (+,+,+)
   to[7-off] = HepPoint3D(  dz*tthcp - h2*ta2 + b2, to[4].y()      ,  dz ); // (+,-,+)

   for( unsigned int i ( 0 ) ; i != 8 ; ++i )
   {
      ko[i] = tr * to[i] ; // apply transformation
   }

   // if reflection, different things for barrel and endcap
   static const HepVector3D x ( 1, 0, 0 ) ;
   static const HepVector3D y ( 0, 1, 0 ) ;
   static const HepVector3D z ( 0, 0, 1 ) ;
   const bool refl ( ( ( tr*x ).cross( tr*y ) ).dot( tr*z ) < 0 ) ; // has reflection!
   if( refl || 
       h1>h2  )
   {
      if( 11.2 < dz ) //barrel
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
      const HepPoint3D& p ( ko[i] ) ;
      co[ i ] = GlobalPoint( p.x(), p.y(), p.z() ) ;
   }
}
//----------------------------------------------------------------------

bool 
TruncatedPyramid::inside( const GlobalPoint& point ) const
{
   bool ans ( false ) ;
   const HepPoint3D p ( point.x(), point.y(), point.z() ) ;
   const CornersVec& cog ( getCorners() ) ;
   HepPoint3D co[8] ;
   for( unsigned int i ( 0 ) ; i != 8 ; ++i )
   {
      co[i] = HepPoint3D( cog[i].x(), cog[i].y(), cog[i].z() ) ;
   }

   const HepPlane3D AA ( co[0], co[1], co[2] ) ; // z<0
   const HepPlane3D BB ( co[6], co[5], co[4] ) ; // z>0
   if( ( p - AA.point(p) ).dot( p - BB.point(p) ) <= 0 )
   {
      const HepPlane3D CC ( co[0], co[4], co[5] ) ; // x<0
      const HepPlane3D DD ( co[2], co[6], co[7] ) ; // x>0
      if( ( p - CC.point(p) ).dot( p - DD.point(p) ) <= 0 )
      {
	 const HepPlane3D EE ( co[0], co[3], co[7] ) ; // y<0
	 const HepPlane3D FF ( co[1], co[5], co[6] ) ; // y>0
	 if( ( p - EE.point(p) ).dot( p - FF.point(p) ) <= 0 )
	 {
	    ans = true ;
	 }
      }
   }
   return ans ;
}

/*   if( 0 == m_bou )
   {
      const CornersVec& cog ( getCorners() ) ;
      std::vector<HepPoint3D> co ( 8, HepPoint3D(0,0,0) ) ;
      for( unsigned int i ( 0 ) ; i != 8 ; ++i )
      {
	 co[i] = HepPoint3D( cog[i].x(), cog[i].y(), cog[i].z() ) ;
      }
      m_bou = new BoundaryVec ;
      m_bou->reserve( 6 ) ; // order is important for this function!
      m_bou->push_back( HepPlane3D( co[0], co[1], co[2] ) ) ; // z<0
      m_bou->push_back( HepPlane3D( co[6], co[5], co[4] ) ) ; // z>0
      m_bou->push_back( HepPlane3D( co[0], co[4], co[5] ) ) ; // x<0
      m_bou->push_back( HepPlane3D( co[2], co[6], co[7] ) ) ; // x>0
      m_bou->push_back( HepPlane3D( co[0], co[3], co[7] ) ) ; // y<0
      m_bou->push_back( HepPlane3D( co[1], co[5], co[6] ) ) ; // y>0
   }
   const BoundaryVec& b ( *m_bou ) ;

   const HepPoint3D p ( point.x(), point.y(), point.z() ) ;

   return ( ( p - b[0].point(p) ).dot( p - b[1].point(p) ) <= 0 &&
	    ( p - b[2].point(p) ).dot( p - b[3].point(p) ) <= 0 &&
	    ( p - b[4].point(p) ).dot( p - b[5].point(p) ) <= 0    ) ;
	    }*/
/*
void TruncatedPyramid::dump( const char * prefix ) const 
{
   std::cout << prefix << "Center: " <<  CaloCellGeometry::getPosition() << std::endl;
   const float thetaaxis ( getThetaAxis() ) ;
   const float phiaxis   ( getPhiAxis()   ) ;
   std::cout << prefix << "Axis: " <<  thetaaxis << " " << phiaxis << std::endl ;
   const CaloCellGeometry::CornerVec& corners ( getCorners() ) ;
   for ( unsigned int  ci=0; ci != corners.size(); ++ci ) 
   {
      std::cout << prefix << "Corner: " << corners[ci] << std::endl;
   }
}
*/
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
  
