#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGenericDetId.h"

#include <Math/Transform3D.h>
#include <Math/EulerAngles.h>

typedef CaloCellGeometry::Pt3D     Pt3D     ;
typedef CaloCellGeometry::Pt3DVec  Pt3DVec  ;
typedef CaloCellGeometry::Tr3D     Tr3D     ;

typedef CaloSubdetectorGeometry::CCGFloat CCGFloat ;

CaloSubdetectorGeometry::CaloSubdetectorGeometry() : 
   m_parMgr ( 0 ) ,
   m_cmgr   ( 0 ) ,
   m_sortedIds (false) ,
   m_deltaPhi  ( 0 ) ,
   m_deltaEta  ( 0 )  
{}


CaloSubdetectorGeometry::~CaloSubdetectorGeometry() 
{ 
   delete m_cmgr ;
   delete m_parMgr ; 
   delete m_deltaPhi ;
   delete m_deltaEta ;
}

const std::vector<DetId>& 
CaloSubdetectorGeometry::getValidDetIds( DetId::Detector /*det*/    , 
					 int             /*subdet*/   ) const 
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
   return cellGeomPtr( CaloGenericDetId( id ).denseIndex() ) ;
}

bool 
CaloSubdetectorGeometry::present( const DetId& id ) const 
{
   return ( 0 != getGeometry( id ) ) ;
}

DetId 
CaloSubdetectorGeometry::getClosestCell( const GlobalPoint& r ) const 
{
   const CCGFloat eta ( r.eta() ) ;
   const CCGFloat phi ( r.phi() ) ;
   uint32_t index ( ~0 ) ;
   CCGFloat closest ( 1e9 ) ;

   for( uint32_t i ( 0 ); i != m_validIds.size() ; ++i ) 
   {
      const CaloCellGeometry* cell ( getGeometry( m_validIds[ i ] ) ) ;
      if( 0 != cell )
      {
	 const GlobalPoint& p ( cell->getPosition() ) ;
	 const CCGFloat eta0 ( p.eta() ) ;
	 const CCGFloat phi0 ( p.phi() ) ;
	 const CCGFloat dR2 ( reco::deltaR2( eta0, phi0, eta, phi ) ) ;
	 if( dR2 < closest ) 
	 {
	    closest = dR2 ;
	    index   = i   ;
	 }
      }
   }
   return ( closest > 0.9e9 ||
	    (uint32_t)(~0) == index       ? DetId(0) :
	    m_validIds[index] ) ;
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
      for( uint32_t i ( 0 ); i != m_validIds.size() ; ++i ) 
      {
	 const CaloCellGeometry* cell ( getGeometry( m_validIds[i] ) ) ;
	 if( 0 != cell )
	 {
	    const GlobalPoint& p ( cell->getPosition() ) ;
	    const CCGFloat eta0 ( p.eta() ) ;
	    if( fabs( eta - eta0 ) < dR )
	    {
	       const CCGFloat phi0 ( p.phi() ) ;
	       CCGFloat delp ( fabs( phi - phi0 ) ) ;
	       if( delp > M_PI ) delp = 2*M_PI - delp ;
	       if( delp < dR )
	       {
		  const CCGFloat dist2 ( reco::deltaR2( eta0, phi0, eta, phi ) ) ;
		  if( dist2 < dR2 ) dss.insert( m_validIds[i] ) ;
	       }
	    }
	 }
      }   
   }
   return dss;
}

CaloSubdetectorGeometry::CellSet 
CaloSubdetectorGeometry::getCellSet( const GlobalPoint& r, double dR ) const {
  // stupid implementation not to be really used...
  DetIdSet ids = getCells(r, dR);
  CellSet cells; cells.reserve(ids.size());
  for ( auto id : ids) cells.push_back(getGeometry(id));
  return cells;
}

void 
CaloSubdetectorGeometry::allocateCorners( CaloCellGeometry::CornersVec::size_type n )
{
   assert( 0 == m_cmgr ) ;
   m_cmgr = new CaloCellGeometry::CornersMgr( n*( CaloCellGeometry::k_cornerSize ),
					      CaloCellGeometry::k_cornerSize        ) ; 

   m_validIds.reserve( n ) ;
}

void 
CaloSubdetectorGeometry::allocatePar( ParVec::size_type n,
				      unsigned int      m     )
{
   assert( 0 == m_parMgr ) ;
   m_parMgr = new ParMgr( n*m, m ) ;
}

void
CaloSubdetectorGeometry::getSummary( CaloSubdetectorGeometry::TrVec&  tVec ,
				     CaloSubdetectorGeometry::IVec&   iVec ,   
				     CaloSubdetectorGeometry::DimVec& dVec   )  const
{
   tVec.reserve( m_validIds.size()*numberOfTransformParms() ) ;
   iVec.reserve( numberOfShapes()==1 ? 1 : m_validIds.size() ) ;
   dVec.reserve( numberOfShapes()*numberOfParametersPerShape() ) ;

   for( ParVecVec::const_iterator ivv ( parVecVec().begin() ) ; ivv != parVecVec().end() ; ++ivv )
   {
      const ParVec& pv ( *ivv ) ;
      for( ParVec::const_iterator iv ( pv.begin() ) ; iv != pv.end() ; ++iv )
      {
	 dVec.push_back( *iv ) ;
      }
   }

   for( uint32_t i ( 0 ) ; i != m_validIds.size() ; ++i )
   {
      Tr3D tr ;
      const CaloCellGeometry* ptr ( cellGeomPtr( i ) ) ;
      assert( 0 != ptr ) ;
      ptr->getTransform( tr, ( Pt3DVec* ) 0 ) ;

      if( Tr3D() == tr ) // for preshower there is no rotation
      {
	 const GlobalPoint& gp ( ptr->getPosition() ) ; 
	 tr = HepGeom::Translate3D( gp.x(), gp.y(), gp.z() ) ;
      }

      const CLHEP::Hep3Vector  tt ( tr.getTranslation() ) ;
      tVec.push_back( tt.x() ) ;
      tVec.push_back( tt.y() ) ;
      tVec.push_back( tt.z() ) ;
      if( 6 == numberOfTransformParms() )
      {
	 const CLHEP::HepRotation rr ( tr.getRotation() ) ;
	 const ROOT::Math::Transform3D rtr ( rr.xx(), rr.xy(), rr.xz(), tt.x(),
					     rr.yx(), rr.yy(), rr.yz(), tt.y(),
					     rr.zx(), rr.zy(), rr.zz(), tt.z()  ) ;
	 ROOT::Math::EulerAngles ea ;
	 rtr.GetRotation( ea ) ;
	 tVec.push_back( ea.Phi() ) ;
	 tVec.push_back( ea.Theta() ) ;
	 tVec.push_back( ea.Psi() ) ;
      }

      const CCGFloat* par ( ptr->param() ) ;

      unsigned int ishape ( 9999 ) ;
      for( unsigned int ivv ( 0 ) ; ivv != parVecVec().size() ; ++ivv )
      {
	 bool ok ( true ) ;
	 const CCGFloat* pv ( &(*parVecVec()[ivv].begin() ) ) ;
	 for( unsigned int k ( 0 ) ; k != numberOfParametersPerShape() ; ++k )
	 {
	    ok = ok && ( fabs( par[k] - pv[k] ) < 1.e-6 ) ;
	 }
	 if( ok ) 
	 {
	    ishape = ivv ;
	    break ;
	 }
      }
      assert( 9999 != ishape ) ;

      const unsigned int nn (( numberOfShapes()==1) ? (unsigned int)1 : m_validIds.size() ) ; 
      if( iVec.size() < nn ) iVec.push_back( ishape ) ;
   }
}

CCGFloat
CaloSubdetectorGeometry::deltaPhi( const DetId& detId ) const
{
   const CaloGenericDetId cgId ( detId ) ;

   if( 0 == m_deltaPhi )
   {
      const uint32_t kSize ( cgId.sizeForDenseIndexing() ) ;
      m_deltaPhi = new std::vector<CCGFloat> ( kSize ) ;
      for( uint32_t i ( 0 ) ; i != kSize ; ++i )
      {
	 const CaloCellGeometry* cellPtr ( cellGeomPtr( i ) ) ;
	 if( 0 != cellPtr )
	 {
	    const CaloCellGeometry& cell ( *cellPtr ) ;
	    CCGFloat dPhi1 ( fabs(
				GlobalPoint( ( cell.getCorners()[0].x() + 
					       cell.getCorners()[1].x() )/2. ,
					     ( cell.getCorners()[0].y() + 
					       cell.getCorners()[1].y() )/2. ,
					     ( cell.getCorners()[0].z() + 
					       cell.getCorners()[1].z() )/2.  ).phi() -
				GlobalPoint( ( cell.getCorners()[2].x() + 
					       cell.getCorners()[3].x() )/2. ,
					     ( cell.getCorners()[2].y() + 
					       cell.getCorners()[3].y() )/2. ,
					     ( cell.getCorners()[2].z() + 
					       cell.getCorners()[3].z() )/2.  ).phi() ) ) ;
	    CCGFloat dPhi2 ( fabs(
				GlobalPoint( ( cell.getCorners()[0].x() + 
					       cell.getCorners()[3].x() )/2. ,
					     ( cell.getCorners()[0].y() + 
					       cell.getCorners()[3].y() )/2. ,
					     ( cell.getCorners()[0].z() + 
					       cell.getCorners()[3].z() )/2.  ).phi() -
				GlobalPoint( ( cell.getCorners()[2].x() + 
					       cell.getCorners()[1].x() )/2. ,
					     ( cell.getCorners()[2].y() + 
					       cell.getCorners()[1].y() )/2. ,
					     ( cell.getCorners()[2].z() + 
					       cell.getCorners()[1].z() )/2.  ).phi() ) ) ;
	    if( M_PI < dPhi1 ) dPhi1 = fabs( dPhi1 - 2.*M_PI ) ;
	    if( M_PI < dPhi2 ) dPhi2 = fabs( dPhi2 - 2.*M_PI ) ;
	    (*m_deltaPhi)[i] = dPhi1>dPhi2 ? dPhi1 : dPhi2 ;
	 }
      }
   }
   return (*m_deltaPhi)[ cgId.denseIndex() ] ;
}

CCGFloat 
CaloSubdetectorGeometry::deltaEta( const DetId& detId ) const
{
   const CaloGenericDetId cgId ( detId ) ;

   if( 0 == m_deltaEta )
   {
      const uint32_t kSize ( cgId.sizeForDenseIndexing() ) ;
      m_deltaEta = new std::vector<CCGFloat> ( kSize ) ;
      for( uint32_t i ( 0 ) ; i != kSize ; ++i )
      {
	 const CaloCellGeometry* cellPtr ( cellGeomPtr( i ) ) ;
	 if( 0 != cellPtr )
	 {
	    const CaloCellGeometry& cell ( *cellPtr ) ;
	    const CCGFloat dEta1 ( fabs(
				      GlobalPoint( ( cell.getCorners()[0].x() + 
						     cell.getCorners()[1].x() )/2. ,
						   ( cell.getCorners()[0].y() + 
						     cell.getCorners()[1].y() )/2. ,
						   ( cell.getCorners()[0].z() + 
						     cell.getCorners()[1].z() )/2.  ).eta() -
				      GlobalPoint( ( cell.getCorners()[2].x() + 
						     cell.getCorners()[3].x() )/2. ,
						   ( cell.getCorners()[2].y() + 
						     cell.getCorners()[3].y() )/2. ,
						   ( cell.getCorners()[2].z() + 
						     cell.getCorners()[3].z() )/2.  ).eta() ) ) ;
	    const CCGFloat dEta2 ( fabs(
				      GlobalPoint( ( cell.getCorners()[0].x() + 
						     cell.getCorners()[3].x() )/2. ,
						   ( cell.getCorners()[0].y() + 
						     cell.getCorners()[3].y() )/2. ,
						   ( cell.getCorners()[0].z() + 
						     cell.getCorners()[3].z() )/2.  ).eta() -
				      GlobalPoint( ( cell.getCorners()[2].x() + 
						     cell.getCorners()[1].x() )/2. ,
						   ( cell.getCorners()[2].y() + 
						     cell.getCorners()[1].y() )/2. ,
						   ( cell.getCorners()[2].z() + 
						     cell.getCorners()[1].z() )/2.  ).eta() ) ) ;
	    (*m_deltaEta)[i] = dEta1>dEta2 ? dEta1 : dEta2 ;
	 }
      }
   }
   return (*m_deltaEta)[ cgId.denseIndex() ] ;
}
