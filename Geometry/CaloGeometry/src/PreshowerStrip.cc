#include "Geometry/CaloGeometry/interface/PreshowerStrip.h"
#include <iostream>


const CaloCellGeometry::CornersVec& 
PreshowerStrip::getCorners() const 
{
   const CornersVec& co ( CaloCellGeometry::getCorners() ) ;
   if( co.uninitialized() ) 
   {
      CornersVec& corners ( setCorners() ) ;

      const GlobalPoint& ctr ( getPosition() ) ;
      const float x ( ctr.x() ) ;
      const float y ( ctr.y() ) ;
      const float z ( ctr.z() ) ;

      const double st ( sin(tilt()) ) ;
      const double ct ( cos(tilt()) ) ;

      for( unsigned int ix ( 0 ) ; ix !=2 ; ++ix )
      {
	 const double sx ( 0 == ix ? -1.0 : +1.0 ) ;
	 for( unsigned int iy ( 0 ) ; iy !=2 ; ++iy )
	 {
	    const double sy ( 0 == iy ? -1.0 : +1.0 ) ;
	    for( unsigned int iz ( 0 ) ; iz !=2 ; ++iz )
	    {
	       const double sz ( 0 == iz ? -1.0 : +1.0 ) ;
	       const unsigned int  i ( 4*iz + 2*ix + 
				       ( 1 == ix ? 1-iy : iy )) ;

	       corners[ i ] = GlobalPoint( 
		  dy()>dx() ? 
		  x + sx*dx() : 
		  x + sx*dx()*ct - sz*dz()*st ,
		  dy()<dx() ? 
		  y + sy*dy() : 
		  y + sy*dy()*ct - sz*dz()*st ,
		  dy()>dx() ? 
		  z + sz*dz()*ct + sy*dy()*st :
		  z + sz*dz()*ct + sx*dx()*st ) ;
	    }
	 }
      }
   }
   return co ;
}

std::ostream& operator<<( std::ostream& s, const PreshowerStrip& cell ) 
{
   s << "Center: " <<  cell.getPosition() << std::endl ;
   s << "dx = " << cell.dx() << ", dy = " << cell.dy() << ", dz = " << cell.dz() << std::endl ;
/*   const CaloCellGeometry::CornerVec& corners ( cell.getCorners() ) ; 
   for( unsigned int ci ( 0 ) ; ci != corners.size(); ci++ ) 
   {
      s  << "Corner: " << corners[ci] << std::endl;
   }*/
   return s;
}

std::vector<HepGeom::Point3D<double> >
PreshowerStrip::localCorners( const double* pv,
			      HepGeom::Point3D<double> &   ref )
{
   assert( 0 != pv ) ;

   const double dx ( pv[0] ) ;
   const double dy ( pv[1] ) ;
   const double dz ( pv[2] ) ;

   std::vector<HepGeom::Point3D<double> > lc ( 8, HepGeom::Point3D<double> (0,0,0) ) ;

   lc[0] = HepGeom::Point3D<double> ( -dx, -dy, -dz ) ;
   lc[1] = HepGeom::Point3D<double> ( -dx,  dy, -dz ) ;
   lc[2] = HepGeom::Point3D<double> (  dx,  dy, -dz ) ;
   lc[3] = HepGeom::Point3D<double> (  dx, -dy, -dz ) ;
   lc[4] = HepGeom::Point3D<double> ( -dx, -dy,  dz ) ;
   lc[5] = HepGeom::Point3D<double> ( -dx,  dy,  dz ) ;
   lc[6] = HepGeom::Point3D<double> (  dx,  dy,  dz ) ;
   lc[7] = HepGeom::Point3D<double> (  dx, -dy,  dz ) ;

   ref   = HepGeom::Point3D<double> (0,0,0) ;

   return lc ;
}
