#include "Geometry/CaloGeometry/interface/PreshowerStrip.h"
#include <iostream>

using namespace std;

bool
PreshowerStrip::inside( const GlobalPoint& p ) const
{
   const GlobalPoint& c ( getPosition() ) ;
   return ( fabs( p.x() - c.x() ) < dx() && 
	    fabs( p.y() - c.y() ) < dy() &&
	    fabs( p.z() - c.z() ) < dz()    ) ; 
}

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

      corners[ 0 ] = GlobalPoint( x - dx(), y - dy(), z - dz() ) ;
      corners[ 1 ] = GlobalPoint( x - dx(), y + dy(), z - dz() ) ;
      corners[ 2 ] = GlobalPoint( x + dx(), y + dy(), z - dz() ) ;
      corners[ 3 ] = GlobalPoint( x + dx(), y - dy(), z - dz() ) ;
      corners[ 4 ] = GlobalPoint( x - dx(), y - dy(), z + dz() ) ;
      corners[ 5 ] = GlobalPoint( x - dx(), y + dy(), z + dz() ) ;
      corners[ 6 ] = GlobalPoint( x + dx(), y + dy(), z + dz() ) ;
      corners[ 7 ] = GlobalPoint( x + dx(), y - dy(), z + dz() ) ;
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
