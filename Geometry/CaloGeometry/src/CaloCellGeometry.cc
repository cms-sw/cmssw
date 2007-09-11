#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"

const CaloCellGeometry::CornersVec&
CaloCellGeometry::getCorners() const
{
   return m_corners ;
}

std::ostream& operator<<( std::ostream& s, const CaloCellGeometry& cell ) 
{
   s << ", Center: " <<  cell.getPosition() << std::endl;

   if( cell.emptyCorners() )
   {
      s << "Corners vector is empty." << std::endl ;
   }
   else
   {
      const CaloCellGeometry::CornersVec& corners ( cell.getCorners() ) ;
      for ( unsigned int i ( 0 ) ; i != corners.size() ; ++i ) 
      {
	 s << "Corner: " << corners[ i ] << std::endl;
      }
   }
   return s ;
}

const float* 
CaloCellGeometry::getParmPtr(
   const std::vector<double>& vd ,
   const unsigned int         np ,
   CaloCellGeometry::ParVecVec& pvv  )
{
   const float* pP ( 0 ) ;

   ParVec vv ;
   vv.reserve( np ) ;

   for( unsigned int i ( 0 ) ; i != vd.size() ; ++i )
   {
      vv.push_back( vd[i] ) ;
   }
   for( unsigned int ii ( 0 ) ; ii != pvv.size() ; ++ii )
   {
      const ParVec& v ( pvv[ii] ) ;
      assert( v.size() == np ) ;

      bool same ( true ) ;
      for( unsigned int j ( 0 ) ; j != np ; ++j )
      {
	 same = same && ( vv[j] == v[j] ) ;
      }

      if( same )
      {
	 pP = &(*v.begin()) ;
	 break ;
      }
   }
   if( 0 == pP )
   {
      pvv.push_back( vv ) ;
      pP = &(*pvv.back().begin()) ;
   }
   return pP ;
}
