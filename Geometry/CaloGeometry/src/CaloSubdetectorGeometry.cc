#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

CaloSubdetectorGeometry::~CaloSubdetectorGeometry() 
{ 
   for( CellCont::iterator i ( m_cellG.begin() );
	i!=m_cellG.end(); ++i )
   {
      delete const_cast<CaloCellGeometry*>((*i).second) ;
   }

   delete m_cmgr ; // must delete after geometries!
}

void 
CaloSubdetectorGeometry::addCell( const DetId& id, 
				  const CaloCellGeometry* ccg )
{
   m_cellG.insert( std::make_pair( id, ccg ) ) ;
}

const CaloCellGeometry* 
CaloSubdetectorGeometry::getGeometry( const DetId& id ) const
{
   CellCont::const_iterator i ( m_cellG.find( id ) ) ;
   return ( i == m_cellG.end() ? 0 : i->second ) ;
}

bool 
CaloSubdetectorGeometry::present( const DetId& id ) const 
{
   return m_cellG.find( id ) != m_cellG.end() ;
}


const std::vector<DetId>& 
CaloSubdetectorGeometry::getValidDetIds( DetId::Detector det,
					 int             subdet ) const 
{
   if( m_validIds.empty() ) 
   {
      m_validIds.reserve( m_cellG.size() ) ;
      for( CellCont::const_iterator i ( cellGeometries().begin() ); 
	   i != cellGeometries().end() ; ++i )
      {
	 m_validIds.push_back(i->first);
      }
      std::sort( m_validIds.begin(), m_validIds.end() ) ;
   }
   return m_validIds ;    
}

double 
CaloSubdetectorGeometry::deltaR( const GlobalPoint& p1, 
				 const GlobalPoint& p2) 
{
   double dp=p1.phi()-p2.phi();
   double de=p1.eta()-p2.eta();
   return std::sqrt(dp*dp+de*de);
}

DetId 
CaloSubdetectorGeometry::getClosestCell( const GlobalPoint& r ) const 
{
   double closest=1e5;
   DetId retval(0);
   for( CellCont::const_iterator i ( m_cellG.begin() ); 
	i != m_cellG.end() ; ++i ) 
   {
      const double dR ( deltaR( r, i->second->getPosition() ) ) ;
      if( dR<closest ) 
      {
	 closest=dR;
	 retval=i->first;
      }
   }   
   return retval;
}


CaloSubdetectorGeometry::DetIdSet 
CaloSubdetectorGeometry::getCells( const GlobalPoint& r, 
				   double dR             ) const 
{
   DetIdSet dss;

   for( CellCont::const_iterator i ( m_cellG.begin() ); 
	i != m_cellG.end() ; ++i ) 
   {
      const double dist ( deltaR( r, i->second->getPosition() ) );
      if( dist <= dR ) dss.insert( i->first ) ;
  }   

  return dss;
}

void 
CaloSubdetectorGeometry::allocateCorners( CaloCellGeometry::CornersVec::size_type n )
{
   assert( 0 == m_cmgr ) ;
   m_cmgr = new CaloCellGeometry::CornersMgr( n*( CaloCellGeometry::k_cornerSize ) ) ; 
}
