#include "Geometry/ForwardGeometry/interface/IdealZDCTrapezoid.h"
#include <math.h>

namespace calogeom {

   std::vector<HepPoint3D>
   IdealZDCTrapezoid::localCorners( const double* pv  ,
				    HepPoint3D&   ref   )
   {
      assert( 0 != pv ) ;

      const double an ( pv[0] ) ;
      const double dx ( pv[1] ) ;
      const double dy ( pv[2] ) ;
      const double dz ( pv[3] ) ;
      const double ta ( tan( an ) ) ;
      const double dt ( dy*ta ) ;

      std::vector<HepPoint3D>  lc ( 8, HepPoint3D( 0,0,0) ) ;

      lc[0] = HepPoint3D( -dx, -dy, +dz+dt ) ;
      lc[1] = HepPoint3D( -dx, +dy, +dz-dt ) ;
      lc[2] = HepPoint3D( +dx, +dy, +dz-dt ) ;
      lc[3] = HepPoint3D( +dx, -dy, +dz+dt ) ;
      lc[4] = HepPoint3D( -dx, -dy, -dz+dt ) ;
      lc[5] = HepPoint3D( -dx, +dy, -dz-dt ) ;
      lc[6] = HepPoint3D( +dx, +dy, -dz-dt ) ;
      lc[7] = HepPoint3D( +dx, -dy, -dz+dt ) ;

      ref   = 0.25*( lc[0] + lc[1] + lc[2] + lc[3] ) ;
      return lc ;
   }

   const CaloCellGeometry::CornersVec& 
   IdealZDCTrapezoid::getCorners() const 
   {
      const CornersVec& co ( CaloCellGeometry::getCorners() ) ;
      if( co.uninitialized() ) 
      {
	 CaloCellGeometry::CornersVec& corners ( setCorners() ) ;
	 const GlobalPoint& p ( getPosition() ) ;
	 const double zsign ( 0 < p.z() ? 1. : -1. ) ;
	 const HepPoint3D gf ( p.x(), p.y(), p.z() ) ;

	 HepPoint3D lf ;
	 const std::vector<HepPoint3D> lc ( localCorners( param(), lf ) ) ;
	 const HepPoint3D lb ( lf.x() , lf.y() , lf.z() - 2.*dz() ) ;
	 const HepPoint3D ls ( lf.x() - dx(), lf.y(), lf.z() ) ;

	 const HepPoint3D  gb ( gf.x() , gf.y() , gf.z() + 2.*zsign*dz() ) ;

	 const HepPoint3D gs ( gf.x() - zsign*dx(),
			       gf.y() ,
			       gf.z()         ) ;

	 const HepTransform3D tr ( lf, lb, ls,
				   gf, gb, gs ) ;

	 for( unsigned int i ( 0 ) ; i != 8 ; ++i )
	 {
	    const HepPoint3D gl ( tr*lc[i] ) ;
	    corners[i] = GlobalPoint( gl.x(), gl.y(), gl.z() ) ;
	 }
      }
      return co ;
   }

   bool 
   IdealZDCTrapezoid::inside( const GlobalPoint& point ) const 
   {
      const HepPoint3D p ( point.x(), point.y(), point.z() ) ;

      bool ok ( false ) ;

      const GlobalPoint& face ( getPosition() ) ;

      if( fabs( p.x() - face.x() ) <= dx() &&
	  fabs( p.y() - face.y() ) <= dy() &&
	  fabs( p.z() - face.z() ) <= 2.*dz() + dt() )
      {
	 const float sign ( 0 < point.z() ? 1 : -1 ) ;
	 const float sl   ( sign*tan( an() ) ) ;
	 ok = ok && ( ( p.z() - face.z() )>= sl*p.y() ) ;
	 ok = ok && ( ( p.z() - ( face.z() + sign*2.*dz() ) ) <= sl*p.y() ) ;
      }
      return ok ;
   }

   std::ostream& operator<<( std::ostream& s, const IdealZDCTrapezoid& cell ) 
   {
      s << "Center: " <<  cell.getPosition() << std::endl ;
      s << "TiltAngle = " << cell.an()*180./M_PI << " deg, dx = " 
	<< cell.dx() 
	<< ", dy = " << cell.dy() << ", dz = " << cell.dz() << std::endl ;
      return s;
   }
}
