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

const double* 
CaloCellGeometry::checkParmPtr(
   const std::vector<double>&   vv  ,
   CaloCellGeometry::ParVecVec& pvv  )
{
   const double* pP ( 0 ) ;

   for( unsigned int ii ( 0 ) ; ii != pvv.size() ; ++ii )
   {
      const ParVec& v ( pvv[ii] ) ;
      assert( v.size() == vv.size() ) ;

      bool same ( true ) ;
      for( unsigned int j ( 0 ) ; j != vv.size() ; ++j )
      {
	 same = same && ( vv[j] == v[j] ) ;
	 if( !same ) break ;
      }
      if( same )
      {
	 pP = &(*v.begin()) ;
	 break ;
      }
   }
   return pP ;
}

const double* 
CaloCellGeometry::getParmPtr(
   const std::vector<double>&   vv  ,
   CaloCellGeometry::ParMgr*    mgr ,
   CaloCellGeometry::ParVecVec& pvv  )
{
   const double* pP ( checkParmPtr( vv, pvv ) ) ;

   if( 0 == pP )
   {
      pvv.push_back( ParVec( mgr ) ) ;
      ParVec& back ( pvv.back() ) ;
      for( unsigned int i ( 0 ) ; i != vv.size() ; ++i )
      {
	 back[i] = vv[i] ;
      }
      pP = &(*back.begin()) ;
   }
   return pP ;
}
