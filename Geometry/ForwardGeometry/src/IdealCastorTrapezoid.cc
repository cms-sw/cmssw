#include "Geometry/ForwardGeometry/interface/IdealCastorTrapezoid.h"
#include "CLHEP/Geometry/Plane3D.h"
#include "CLHEP/Geometry/Transform3D.h"
#include <math.h>

namespace calogeom {

   std::vector<HepPoint3D>
   IdealCastorTrapezoid::localCorners( const double* pv  ,
				       HepPoint3D&   ref   )
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

      std::vector<HepPoint3D>  lc ( 8, HepPoint3D( 0,0,0) ) ;

      lc[ 0 ] = HepPoint3D(        -dx, -dy ,  dzb ) ;
      lc[ 1 ] = HepPoint3D(        -dx, +dy ,  dzs ) ;
      lc[ 2 ] = HepPoint3D( +2*dxh -dx, +dy ,  dzs ) ;
      lc[ 3 ] = HepPoint3D( +2*dxl -dx, -dy ,  dzb ) ;
      lc[ 4 ] = HepPoint3D(        -dx, -dy , -dzs ) ;
      lc[ 5 ] = HepPoint3D(        -dx, +dy , -dzb ) ;
      lc[ 6 ] = HepPoint3D( +2*dxh -dx, +dy , -dzb ) ;
      lc[ 7 ] = HepPoint3D( +2*dxl -dx, -dy , -dzs ) ;

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
	 const HepPoint3D gf ( p.x(), p.y(), p.z() ) ;

	 HepPoint3D lf ;
	 const std::vector<HepPoint3D> lc ( localCorners( param(), lf ) ) ;
	 const HepPoint3D lb ( lf.x() , lf.y() , lf.z() - 2.*dz() ) ;
	 const HepPoint3D ls ( lf.x() - dx(), lf.y(), lf.z() ) ;


	 const double fphi ( atan( dx()/( dR() + dy() ) ) ) ;
	 const HepPoint3D  gb ( gf.x() , gf.y() , gf.z() + 2.*zsign*dz() ) ;

	 const double rho ( dR() + dy() ) ;
	 const double phi ( gf.phi() + fphi ) ;
	 const HepPoint3D gs ( rho*cos(phi) ,
			       rho*sin(phi) ,
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
   IdealCastorTrapezoid::inside( const GlobalPoint& point ) const 
   {
      const HepPoint3D p ( point.x(), point.y(), point.z() ) ;

      bool ok ( false ) ;

      // loose cut to avoid some calculations
      const HepPoint3D fc ( getPosition().x(),
			    getPosition().y(),
			    getPosition().z() ) ;

      if( 0 < p.z()*fc.z()                                 && //gross cuts
	  fabs( p.z() ) > fabs( fc.z() ) - dhz()           &&
	  fabs( p.z() ) < fabs( fc.z() ) + 2.*dz() + dhz() &&
	  fabs( p.perp() ) > dR()                          &&
	  fabs( p.perp() ) < dR() + 3.*dy()                  )
      {
	 ok = true ;

	 const CaloCellGeometry::CornersVec& gc ( getCorners() ) ;
	 const HepPoint3D cv[8] = 
	    { HepPoint3D( gc[0].x(), gc[0].y(), gc[0].z() ) ,
	      HepPoint3D( gc[1].x(), gc[1].y(), gc[1].z() ) ,
	      HepPoint3D( gc[2].x(), gc[2].y(), gc[2].z() ) ,
	      HepPoint3D( gc[3].x(), gc[3].y(), gc[3].z() ) ,
	      HepPoint3D( gc[4].x(), gc[4].y(), gc[4].z() ) ,
	      HepPoint3D( gc[5].x(), gc[5].y(), gc[5].z() ) ,
	      HepPoint3D( gc[6].x(), gc[6].y(), gc[6].z() ) ,
	      HepPoint3D( gc[7].x(), gc[7].y(), gc[7].z() ) } ;

	 for( unsigned int face ( 0 ) ; face != 6 ; ++(++face) )
	 {
	    static const unsigned int nc[6][4] = 
	       { { 0,1,2,3 }, { 7,6,5,4 }, 
		 { 0,4,5,1 }, { 3,2,6,7 },
		 { 0,3,7,4 }, { 1,5,6,2 } } ;
	    const unsigned int* ic1 ( &nc[face  ][0] ) ;
	    const unsigned int* ic2 ( &nc[face+1][0] ) ;
	    const HepPlane3D pl1 ( cv[ic1[0]], cv[ic1[1]], cv[ic1[2]] ) ;
	    const HepPlane3D pl2 ( cv[ic2[0]], cv[ic2[1]], cv[ic2[2]] ) ;

/*	    const HepPoint3D p1 ( pl1.point( p ) ) ;
	    const HepPoint3D p2 ( pl2.point( p ) ) ;
	    const HepVector3D v1 ( p1 - p ) ;
	    const HepVector3D v2 ( p2 - p ) ;*/

	    if( 0 > pl1.distance(p)*pl2.distance(p) ) //v1.dot( v2 ) )
	    {
	       ok = false ;
	       break ;
	    }
	 }
      }
      return ok ;
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
