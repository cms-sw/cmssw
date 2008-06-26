#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGenericDetId.h"

CaloSubdetectorGeometry::~CaloSubdetectorGeometry() 
{ 
   for( CellCont::iterator i ( m_cellG.begin() );
	i!=m_cellG.end(); ++i )
   {
      delete *i ;
   }

   delete m_cmgr ; // must delete *after* geometries!
   delete m_parMgr ; 
}

void 
CaloSubdetectorGeometry::addCell( const DetId&            id  , 
				  const CaloCellGeometry* ccg   )
{
   const CaloGenericDetId cdid ( id ) ;
/*   if( cdid.validDetId() )
   {
*/
  const uint32_t index ( cdid.denseIndex() ) ;
/*      if( cdid.rawId() == CaloGenericDetId( cdid.det(), 
					    cdid.subdetId(),
					    index           ) ) // double check all is ok
      {
	 if( index >= m_cellG.size() ) std::cout<<" Index ="<< index<< ", but len = "<<m_cellG.size() <<std::endl ;
*/
	 m_cellG[    index ] = ccg ;
	 m_validIds.push_back( id )  ;
/*      }
      else
      {
	 std::cout<<"Bad index in CaloSubdetectorGeometry.cc: "<< index 
		  <<", id="<<cdid<< std::endl ;
      }
   }
   else
   {
      std::cout<<"Bad id in CaloSubdetectorGeometry.cc: "<<cdid<<std::endl ;
      }*/
}

const std::vector<DetId>& 
CaloSubdetectorGeometry::getValidDetIds( DetId::Detector det    , 
					 int             subdet   ) const 
{
   if( !m_sortedIds )
   {
      m_sortedIds = true ;
      std::sort( m_validIds.begin(), m_validIds.end() ) ;
   }
   return m_validIds ;
}

const CaloCellGeometry* 
CaloSubdetectorGeometry::getGeometry( const DetId& id ) const
{
   return m_cellG[ CaloGenericDetId( id ).denseIndex() ] ;
}

bool 
CaloSubdetectorGeometry::present( const DetId& id ) const 
{
//   return m_cellG.find( id ) != m_cellG.end() ;
   return 0 != m_cellG[ CaloGenericDetId( id ).denseIndex() ] ;
}

DetId 
CaloSubdetectorGeometry::getClosestCell( const GlobalPoint& r ) const 
{
   const double eta ( r.eta() ) ;
   const double phi ( r.phi() ) ;
   uint32_t index ( ~0 ) ;
   double closest ( 1e9 ) ;

   CellCont::const_iterator cBeg ( cellGeometries().begin() ) ;
   for( CellCont::const_iterator i ( cBeg ); 
	i != m_cellG.end() ; ++i ) 
   {
      const GlobalPoint& p ( (*i)->getPosition() ) ;
      const double eta0 ( p.eta() ) ;
      const double phi0 ( p.phi() ) ;
      const double dR2 ( reco::deltaR2( eta0, phi0, eta, phi ) ) ;
      if( dR2 < closest ) 
      {
	 closest = dR2 ;
	 index   = i - cBeg ;
      }
   }   
   const DetId tid ( m_validIds.front() ) ;
   return ( closest > 0.9e9 ? DetId(0) : CaloGenericDetId( tid.det(),
							   tid.subdetId(),
							   index           ) ) ;
}

CaloSubdetectorGeometry::DetIdSet 
CaloSubdetectorGeometry::getCells( const GlobalPoint& r, 
				   double dR             ) const 
{
   const double dR2 ( dR*dR ) ;
   const double eta ( r.eta() ) ;
   const double phi ( r.phi() ) ;

   DetIdSet dss;
   
   CellCont::const_iterator cBeg ( cellGeometries().begin() ) ;
   if( 0.000001 < dR )
   {
      for( CellCont::const_iterator i ( m_cellG.begin() ); 
	   i != m_cellG.end() ; ++i ) 
      {
	 const GlobalPoint& p ( (*i)->getPosition() ) ;
	 const double eta0 ( p.eta() ) ;
	 if( fabs( eta - eta0 ) < dR )
	 {
	    const double phi0 ( p.phi() ) ;
	    double delp ( fabs( phi - phi0 ) ) ;
	    if( delp > M_PI ) delp = 2*M_PI - delp ;
	    if( delp < dR )
	    {
	       const double dist2 ( reco::deltaR2( eta0, phi0, eta, phi ) ) ;
	       const DetId tid ( m_validIds.front() ) ;
	       if( dist2 < dR2 ) dss.insert( CaloGenericDetId( tid.det(),
							       tid.subdetId(),
							       i - cBeg )     ) ;
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

   m_validIds.reserve( n ) ;

   m_cellG.reserve(    n ) ;
   m_cellG.assign(     n, CellCont::value_type ( 0 ) ) ;
}

void 
CaloSubdetectorGeometry::allocatePar( ParVec::size_type n,
				      unsigned int      m     )
{
   assert( 0 == m_parMgr ) ;
   m_parMgr = new ParMgr( n*m, m ) ;
}

