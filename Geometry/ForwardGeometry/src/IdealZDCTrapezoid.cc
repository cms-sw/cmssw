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

   std::ostream& operator<<( std::ostream& s, const IdealZDCTrapezoid& cell ) 
   {
      s << "Center: " <<  cell.getPosition() << std::endl ;
      s << "TiltAngle = " << cell.an()*180./M_PI << " deg, dx = " 
	<< cell.dx() 
	<< ", dy = " << cell.dy() << ", dz = " << cell.dz() << std::endl ;
      return s;
   }
}
