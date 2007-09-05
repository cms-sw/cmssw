#include "Geometry/CaloGeometry/interface/PreshowerStrip.h"
#include <iostream>

using namespace std;

bool
PreshowerStrip::inside( const GlobalPoint& p ) const
{
   const GlobalPoint& c ( getPosition() ) ;
   return ( fabs( p.x() - c.x() ) < m_dx && 
	    fabs( p.y() - c.y() ) < m_dy &&
	    fabs( p.z() - c.z() ) < m_dz    ) ; 
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

      corners[ 0 ] = GlobalPoint( x - m_dx, y - m_dy, z - m_dz ) ;
      corners[ 1 ] = GlobalPoint( x - m_dx, y + m_dy, z - m_dz ) ;
      corners[ 2 ] = GlobalPoint( x + m_dx, y + m_dy, z - m_dz ) ;
      corners[ 3 ] = GlobalPoint( x + m_dx, y - m_dy, z - m_dz ) ;
      corners[ 4 ] = GlobalPoint( x - m_dx, y - m_dy, z + m_dz ) ;
      corners[ 5 ] = GlobalPoint( x - m_dx, y + m_dy, z + m_dz ) ;
      corners[ 6 ] = GlobalPoint( x + m_dx, y + m_dy, z + m_dz ) ;
      corners[ 7 ] = GlobalPoint( x + m_dx, y - m_dy, z + m_dz ) ;
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
