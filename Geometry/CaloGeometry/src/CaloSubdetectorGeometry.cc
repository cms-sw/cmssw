#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

CaloSubdetectorGeometry::~CaloSubdetectorGeometry() 
{ 
   for( CellCont::iterator i ( m_cellG.begin() );
	i!=m_cellG.end(); ++i )
   {
      delete const_cast<CaloCellGeometry*>((*i).second) ;
   }

   delete m_cmgr ; // must delete after geometries!
   delete m_parMgr ; 
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

DetId 
CaloSubdetectorGeometry::getClosestCell( const GlobalPoint& r ) const 
{
   const double eta ( r.eta() ) ;
   const double phi ( r.phi() ) ;
   double closest ( 1e9 ) ;
   DetId retval(0);
   for( CellCont::const_iterator i ( m_cellG.begin() ); 
	i != m_cellG.end() ; ++i ) 
   {
      const GlobalPoint& p ( i->second->getPosition() ) ;
      const double eta0 ( p.eta() ) ;
      const double phi0 ( p.phi() ) ;
      const double dR2 ( reco::deltaR2( eta0, phi0, eta, phi ) ) ;
      if( dR2 < closest ) 
      {
	 closest = dR2 ;
	 retval  = i->first ;
      }
   }   
   return retval;
}

CaloSubdetectorGeometry::DetIdSet 
CaloSubdetectorGeometry::getCells( const GlobalPoint& r, 
				   double dR             ) const 
{
   const double dR2 ( dR*dR ) ;
   const double eta ( r.eta() ) ;
   const double phi ( r.phi() ) ;

   DetIdSet dss;
   
   if( 0.000001 < dR )
   {
      for( CellCont::const_iterator i ( m_cellG.begin() ); 
	   i != m_cellG.end() ; ++i ) 
      {
	 const GlobalPoint& p ( i->second->getPosition() ) ;
	 const double eta0 ( p.eta() ) ;
	 if( fabs( eta - eta0 ) < dR )
	 {
	    const double phi0 ( p.phi() ) ;
	    double delp ( fabs( phi - phi0 ) ) ;
	    if( delp > M_PI ) delp = 2*M_PI - delp ;
	    if( delp < dR )
	    {
	       const double dist2 ( reco::deltaR2( eta0, phi0, eta, phi ) ) ;
	       if( dist2 < dR2 ) dss.insert( i->first ) ;
	    }
	 }
      }   
   }
   return dss;
}

void 
CaloSubdetectorGeometry::allocateCorners( CaloCellGeometry::CornersVec::size_type n )
{
   assert( 0 == m_cmgr ) ;
   m_cmgr = new CaloCellGeometry::CornersMgr( n*( CaloCellGeometry::k_cornerSize ),
					      CaloCellGeometry::k_cornerSize        ) ; 
}

void 
CaloSubdetectorGeometry::allocatePar( ParVec::size_type n,
				      unsigned int      m     )
{
   assert( 0 == m_parMgr ) ;
   m_parMgr = new ParMgr( n*m, m ) ;
}

