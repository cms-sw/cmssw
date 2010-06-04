#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGenericDetId.h"

#include <Math/Transform3D.h>
#include <Math/EulerAngles.h>

using namespace CLHEP;

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
CaloSubdetectorGeometry::addCell( const DetId&      id  , 
				  CaloCellGeometry* ccg   )
{
   const CaloGenericDetId cdid ( id ) ;
/*   if( cdid.validDetId() )
   {
*/
   const uint32_t index ( cdid.denseIndex() ) ;

/*
      if( cdid.rawId() == CaloGenericDetId( cdid.det(), 
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
      if( 0 != *i )
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
   }   
   const DetId tid ( m_validIds.front() ) ;
   return ( closest > 0.9e9 ||
	    (uint32_t)(~0) == index       ? DetId(0) :
	    CaloGenericDetId( tid.det(),
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
	 if( 0 != *i )
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

void
CaloSubdetectorGeometry::getSummary( CaloSubdetectorGeometry::TrVec&  tVec ,
				     CaloSubdetectorGeometry::IVec&   iVec ,   
				     CaloSubdetectorGeometry::DimVec& dVec   )  const
{
   tVec.reserve( cellGeometries().size()*numberOfTransformParms() ) ;
   iVec.reserve( numberOfShapes()==1 ? 1 : cellGeometries().size() ) ;
   dVec.reserve( numberOfShapes()*numberOfParametersPerShape() ) ;

   for( ParVecVec::const_iterator ivv ( parVecVec().begin() ) ; ivv != parVecVec().end() ; ++ivv )
   {
      const ParVec& pv ( *ivv ) ;
      for( ParVec::const_iterator iv ( pv.begin() ) ; iv != pv.end() ; ++iv )
      {
	 dVec.push_back( *iv ) ;
      }
   }

   for( CellCont::const_iterator i ( cellGeometries().begin() ) ; 
	i != cellGeometries().end() ; ++i )
   {
      HepGeom::Transform3D tr ( (*i)->getTransform( ( std::vector<HepGeom::Point3D<double> >* ) 0 ) ) ;

      if( HepGeom::Transform3D() == tr ) // for preshower there is no rotation
      {
	 const GlobalPoint& gp ( (*i)->getPosition() ) ; 
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

      const double* par ( (*i)->param() ) ;

      unsigned int ishape ( 9999 ) ;
      for( unsigned int ivv ( 0 ) ; ivv != parVecVec().size() ; ++ivv )
      {
	 bool ok ( true ) ;
	 const double* pv ( &(*parVecVec()[ivv].begin() ) ) ;
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

      const unsigned int nn (( numberOfShapes()==1) ? (unsigned int)1 : cellGeometries().size() ) ; 
      if( iVec.size() < nn ) iVec.push_back( ishape ) ;
   }
}
