#include "Geometry/ForwardGeometry/interface/IdealCastorTrapezoid.h"
#include "CLHEP/Geometry/Plane3D.h"
#include "CLHEP/Geometry/Transform3D.h"
#include <math.h>

namespace calogeom {

   std::vector<HepGeom::Point3D<double> >
   IdealCastorTrapezoid::localCorners( const double* pv  ,
				       HepGeom::Point3D<double> &   ref   )
   {
      assert( 0 != pv ) ;

      const double dxl ( pv[0] ) ;
      const double dxh ( pv[1] ) ;
      const double dh  ( pv[2] ) ;
      const double dz  ( pv[3] ) ;
      const double an  ( pv[4] ) ;
      const double dx  ( ( dxl +dxh )/2. ) ;
      const double dy  ( dh*sin(an) ) ;
      const double dhz ( dh*cos(an) ) ;
      const double dzb ( dz + dhz ) ;
      const double dzs ( dz - dhz ) ;

      assert( 0 < (dxl*dxh) ) ;

      std::vector<HepGeom::Point3D<double> >  lc ( 8, HepGeom::Point3D<double> ( 0,0,0) ) ;

      lc[ 0 ] = HepGeom::Point3D<double> (        -dx, -dy ,  dzb ) ;
      lc[ 1 ] = HepGeom::Point3D<double> (        -dx, +dy ,  dzs ) ;
      lc[ 2 ] = HepGeom::Point3D<double> ( +2*dxh -dx, +dy ,  dzs ) ;
      lc[ 3 ] = HepGeom::Point3D<double> ( +2*dxl -dx, -dy ,  dzb ) ;
      lc[ 4 ] = HepGeom::Point3D<double> (        -dx, -dy , -dzs ) ;
      lc[ 5 ] = HepGeom::Point3D<double> (        -dx, +dy , -dzb ) ;
      lc[ 6 ] = HepGeom::Point3D<double> ( +2*dxh -dx, +dy , -dzb ) ;
      lc[ 7 ] = HepGeom::Point3D<double> ( +2*dxl -dx, -dy , -dzs ) ;

      ref   = 0.25*( lc[0] + lc[1] + lc[2] + lc[3] ) ;
      return lc ;
   }

   const CaloCellGeometry::CornersVec& 
   IdealCastorTrapezoid::getCorners() const 
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


	 const double fphi ( atan( dx()/( dR() + dy() ) ) ) ;
	 const HepGeom::Point3D<double>   gb ( gf.x() , gf.y() , gf.z() + 2.*zsign*dz() ) ;

	 const double rho ( dR() + dy() ) ;
	 const double phi ( gf.phi() + fphi ) ;
	 const HepGeom::Point3D<double>  gs ( rho*cos(phi) ,
			       rho*sin(phi) ,
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

   std::ostream& operator<<( std::ostream& s, const IdealCastorTrapezoid& cell ) 
   {
      s << "Center: " <<  cell.getPosition() << std::endl ;
//      s 	 << ", dx = " << cell.dx() 
//<< "TiltAngle = " << cell.an() 
//	<< ", dy = " << cell.dy() << ", dz = " << cell.dz() << std::endl ;
      return s;
   }
}
