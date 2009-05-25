#include "Geometry/ForwardGeometry/interface/IdealZDCTrapezoid.h"
#include <math.h>

namespace calogeom {

   std::vector<HepGeom::Point3D<double> >
   IdealZDCTrapezoid::localCorners( const double* pv  ,
				    HepGeom::Point3D<double> &   ref   )
   {
      assert( 0 != pv ) ;

      const double an ( pv[0] ) ;
      const double dx ( pv[1] ) ;
      const double dy ( pv[2] ) ;
      const double dz ( pv[3] ) ;
      const double ta ( tan( an ) ) ;
      const double dt ( dy*ta ) ;

      std::vector<HepGeom::Point3D<double> >  lc ( 8, HepGeom::Point3D<double> ( 0,0,0) ) ;

      lc[0] = HepGeom::Point3D<double> ( -dx, -dy, +dz+dt ) ;
      lc[1] = HepGeom::Point3D<double> ( -dx, +dy, +dz-dt ) ;
      lc[2] = HepGeom::Point3D<double> ( +dx, +dy, +dz-dt ) ;
      lc[3] = HepGeom::Point3D<double> ( +dx, -dy, +dz+dt ) ;
      lc[4] = HepGeom::Point3D<double> ( -dx, -dy, -dz+dt ) ;
      lc[5] = HepGeom::Point3D<double> ( -dx, +dy, -dz-dt ) ;
      lc[6] = HepGeom::Point3D<double> ( +dx, +dy, -dz-dt ) ;
      lc[7] = HepGeom::Point3D<double> ( +dx, -dy, -dz+dt ) ;

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
	 const HepGeom::Point3D<double>  gf ( p.x(), p.y(), p.z() ) ;

	 HepGeom::Point3D<double>  lf ;
	 const std::vector<HepGeom::Point3D<double> > lc ( localCorners( param(), lf ) ) ;
	 const HepGeom::Point3D<double>  lb ( lf.x() , lf.y() , lf.z() - 2.*dz() ) ;
	 const HepGeom::Point3D<double>  ls ( lf.x() - dx(), lf.y(), lf.z() ) ;

	 const HepGeom::Point3D<double>   gb ( gf.x() , gf.y() , gf.z() + 2.*zsign*dz() ) ;

	 const HepGeom::Point3D<double>  gs ( gf.x() - zsign*dx(),
			       gf.y() ,
			       gf.z()         ) ;

	 const HepGeom::Transform3D tr ( lf, lb, ls,
				   gf, gb, gs ) ;

	 for( unsigned int i ( 0 ) ; i != 8 ; ++i )
	 {
	    const HepGeom::Point3D<double>  gl ( tr*lc[i] ) ;
	    corners[i] = GlobalPoint( gl.x(), gl.y(), gl.z() ) ;
	 }
      }
      return co ;
   }

   bool 
   IdealZDCTrapezoid::inside( const GlobalPoint& point ) const 
   {
      const HepGeom::Point3D<double>  p ( point.x(), point.y(), point.z() ) ;

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
