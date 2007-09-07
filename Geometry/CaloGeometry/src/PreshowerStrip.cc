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
   if( co.empty() ) 
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
