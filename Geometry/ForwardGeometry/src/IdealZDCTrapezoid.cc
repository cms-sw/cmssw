#include "Geometry/ForwardGeometry/interface/IdealZDCTrapezoid.h"
#include <math.h>

namespace calogeom {

   const CaloCellGeometry::CornersVec& 
   IdealZDCTrapezoid::getCorners() const 
   {
      const CornersVec& co ( CaloCellGeometry::getCorners() ) ;
      if( co.empty() ) 
      {
	 CaloCellGeometry::CornersVec& corners ( setCorners() ) ;
	 const GlobalPoint& p ( getPosition() ) ;
	 const float zz   ( p.z() ) ;
	 const float yy   ( p.y() ) ;
	 const float cdy  ( cos( m_an )*m_dy/2. ) ;
	 const float sdy  ( sin( m_an )*m_dy/2. ) ;
	 const float sign ( zz<0 ? 1 : -1 ) ;
	 const float z1   ( zz + sign*cdy ) ;
	 const float z2   ( zz - sign*cdy ) ;
	 const float z3   ( z1 + m_dz ) ;
	 const float z4   ( z2 + m_dz ) ;
	 const float x1   (  m_dx/2 );
	 const float x2   ( -m_dx/2 );
	 const float y1   ( yy + sdy ) ;
	 const float y2   ( yy - sdy ) ;
	 corners[ 0 ] = GlobalPoint( x1, y1, z1 ) ; 
	 corners[ 1 ] = GlobalPoint( x2, y1, z1 ) ;
	 corners[ 2 ] = GlobalPoint( x2, y2, z2 ) ;
	 corners[ 3 ] = GlobalPoint( x1, y2, z2 ) ;
	 corners[ 4 ] = GlobalPoint( x1, y1, z3 ) ;
	 corners[ 5 ] = GlobalPoint( x2, y1, z3 ) ;
	 corners[ 6 ] = GlobalPoint( x2, y2, z4 ) ;
	 corners[ 7 ] = GlobalPoint( x1, y2, z4 ) ;
      }
      return co ;
   }

   bool 
   IdealZDCTrapezoid::inside( const GlobalPoint& point ) const 
   {
      bool is_inside ( false ) ;

      const GlobalPoint& face ( getPosition() ) ;

      if( fabs( point.x() - face.x() ) <= m_dx/2   &&
	  fabs( point.y() - face.y() ) <= sin( m_an )*m_dy/2 )
      {
	 const float sl   ( tan( m_an ) ) ;
	 const float blow ( face.y() - sl*face.z() ) ;

	 const float sign ( 0 < point.z() ? 1 : -1 ) ;
	 const float bhigh ( face.y() - sl*sign*( face.z() + sign*m_dz ) ) ;

	 is_inside = ( ( point.z() >  sign*( point.y() - blow  )/sl )  &&
		       ( point.z() <= sign*( point.y() - bhigh )/sl )     ) ;
      }
      return is_inside;
   }

   std::ostream& operator<<( std::ostream& s, const IdealZDCTrapezoid& cell ) 
   {
      s << "Center: " <<  cell.getPosition() << std::endl ;
      s << "TiltAngle = " << cell.an() << ", dx = " << cell.dx() 
	<< ", dy = " << cell.dy() << ", dz = " << cell.dz() << std::endl ;
      return s;
   }
}
