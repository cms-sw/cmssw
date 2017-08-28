#include "Geometry/CaloGeometry/interface/CaloGenericDetId.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <CLHEP/Geometry/Point3D.h>
#include <CLHEP/Geometry/Plane3D.h>

typedef CaloCellGeometry::CCGFloat CCGFloat ;
typedef CaloCellGeometry::Pt3D     Pt3D     ;
typedef CaloCellGeometry::Pt3DVec  Pt3DVec  ;
typedef HepGeom::Plane3D<CCGFloat> Pl3D     ;

EcalEndcapGeometry::EcalEndcapGeometry( void )
  : _nnmods( 316 ),
    _nncrys( 25 ),
    zeP( 0. ),
    zeN( 0. ),
    m_wref( 0. ),
    m_del( 0. ),
    m_nref( 0 ),
    m_borderMgr( nullptr ),
    m_borderPtrVec( nullptr ),
    m_avgZ( -1 ),
    m_check( false ),
    m_cellVec( k_NumberOfCellsForCorners )
{
  m_xlo[0] = 999.;
  m_xlo[1] = 999.;
  m_xhi[0] = -999.;
  m_xhi[1] = -999.;
  m_ylo[0] = 999.;
  m_ylo[1] = 999.;
  m_yhi[0] = -999.;
  m_yhi[1] = -999.;
  m_xoff[0] = 0.;
  m_xoff[1] = 0.;
  m_yoff[0] = 0.;
  m_yoff[1] = 0.;
}

EcalEndcapGeometry::~EcalEndcapGeometry() 
{
  if(m_borderPtrVec)
  {
    auto ptr = m_borderPtrVec.load(std::memory_order_acquire);
    for(auto& v: (*ptr)) {
        delete v;
        v = nullptr;
    }
    delete m_borderPtrVec.load() ;
  }
  delete m_borderMgr.load() ;
}

unsigned int
EcalEndcapGeometry::alignmentTransformIndexLocal( const DetId& id )
{
   const CaloGenericDetId gid ( id ) ;

   assert( gid.isEE() ) ;
   unsigned int index ( EEDetId(id).ix()/51 + ( EEDetId(id).zside()<0 ? 0 : 2 ) ) ;

   return index ;
}

DetId 
EcalEndcapGeometry::detIdFromLocalAlignmentIndex( unsigned int iLoc )
{
   return EEDetId( 20 + 50*( iLoc%2 ), 50, 2*( iLoc/2 ) - 1 ) ;
}

unsigned int
EcalEndcapGeometry::alignmentTransformIndexGlobal( const DetId& /*id*/ )
{
   return (unsigned int)DetId::Ecal - 1 ;
}

void 
EcalEndcapGeometry::initializeParms()
{
  zeP=0.;
  zeN=0.;
  unsigned nP=0;
  unsigned nN=0;
  m_nref = 0 ;

  for( uint32_t i ( 0 ) ; i != m_cellVec.size() ; ++i )
  {
     const CaloCellGeometry* cell ( cellGeomPtr(i) ) ;
     if( nullptr != cell )
     {
	const CCGFloat z ( cell->getPosition().z() ) ;
	if(z>0.)
	{
	   zeP+=z;
	   ++nP;
	}
	else
	{
	   zeN+=z;
	   ++nN;
	}
	const EEDetId myId ( EEDetId::detIdFromDenseIndex(i) ) ;
	const unsigned int ix ( myId.ix() ) ;
	const unsigned int iy ( myId.iy() ) ;
	if( ix > m_nref ) m_nref = ix ;
	if( iy > m_nref ) m_nref = iy ;
     }
  }
  if( 0 < nP ) zeP/=(CCGFloat)nP;
  if( 0 < nN ) zeN/=(CCGFloat)nN;

  m_xlo[0] =  999 ;
  m_xhi[0] = -999 ;
  m_ylo[0] =  999 ;
  m_yhi[0] = -999 ;
  m_xlo[1] =  999 ;
  m_xhi[1] = -999 ;
  m_ylo[1] =  999 ;
  m_yhi[1] = -999 ;
  for( uint32_t i ( 0 ) ; i != m_cellVec.size() ; ++i )
  {
     const CaloCellGeometry* cell ( cellGeomPtr(i) ) ;
     if( nullptr != cell )
     {
	const GlobalPoint& p ( cell->getPosition()  ) ;
	const CCGFloat z ( p.z() ) ;
	const CCGFloat zz ( 0 > z ? zeN : zeP ) ;
	const CCGFloat x ( p.x()*zz/z ) ;
	const CCGFloat y ( p.y()*zz/z ) ;

	if( 0 > z && x < m_xlo[0] ) m_xlo[0] = x ;
	if( 0 < z && x < m_xlo[1] ) m_xlo[1] = x ;
	if( 0 > z && y < m_ylo[0] ) m_ylo[0] = y ;
	if( 0 < z && y < m_ylo[1] ) m_ylo[1] = y ;
     
	if( 0 > z && x > m_xhi[0] ) m_xhi[0] = x ;
	if( 0 < z && x > m_xhi[1] ) m_xhi[1] = x ;
	if( 0 > z && y > m_yhi[0] ) m_yhi[0] = y ;
	if( 0 < z && y > m_yhi[1] ) m_yhi[1] = y ;
     }
  }

  m_xoff[0] = ( m_xhi[0] + m_xlo[0] )/2. ;
  m_xoff[1] = ( m_xhi[1] + m_xlo[1] )/2. ;
  m_yoff[0] = ( m_yhi[0] + m_ylo[0] )/2. ;
  m_yoff[1] = ( m_yhi[1] + m_ylo[1] )/2. ;

  m_del = ( m_xhi[0] - m_xlo[0] + m_xhi[1] - m_xlo[1] +
	    m_yhi[0] - m_ylo[0] + m_yhi[1] - m_ylo[1]   ) ;

  if( 1 != m_nref ) m_wref = m_del/(4.*(m_nref-1)) ;

  m_xlo[0] -= m_wref/2 ;
  m_xlo[1] -= m_wref/2 ;
  m_xhi[0] += m_wref/2 ;
  m_xhi[1] += m_wref/2 ;

  m_ylo[0] -= m_wref/2 ;
  m_ylo[1] -= m_wref/2 ;
  m_yhi[0] += m_wref/2 ;
  m_yhi[1] += m_wref/2 ;

  m_del += m_wref ;
/*
  std::cout<<"zeP="<<zeP<<", zeN="<<zeN<<", nP="<<nP<<", nN="<<nN<<std::endl ;

  std::cout<<"xlo[0]="<<m_xlo[0]<<", xlo[1]="<<m_xlo[1]<<", xhi[0]="<<m_xhi[0]<<", xhi[1]="<<m_xhi[1]
	   <<"\nylo[0]="<<m_ylo[0]<<", ylo[1]="<<m_ylo[1]<<", yhi[0]="<<m_yhi[0]<<", yhi[1]="<<m_yhi[1]<<std::endl ;

  std::cout<<"xoff[0]="<<m_xoff[0]<<", xoff[1]"<<m_xoff[1]<<", yoff[0]="<<m_yoff[0]<<", yoff[1]"<<m_yoff[1]<<std::endl ;

  std::cout<<"nref="<<m_nref<<", m_wref="<<m_wref<<std::endl ;
*/  
}


unsigned int 
EcalEndcapGeometry::xindex( CCGFloat x,
			    CCGFloat z ) const
{
   const CCGFloat xlo ( 0 > z ? m_xlo[0]  : m_xlo[1]  ) ;
   const int i ( 1 + int( ( x - xlo )/m_wref ) ) ;

   return ( 1 > i ? 1 :
	    ( m_nref < (unsigned int) i ? m_nref : (unsigned int) i ) ) ;

}

unsigned int 
EcalEndcapGeometry::yindex( CCGFloat y,
			    CCGFloat z  ) const
{
   const CCGFloat ylo ( 0 > z ? m_ylo[0]  : m_ylo[1]  ) ;
   const int i ( 1 + int( ( y - ylo )/m_wref ) ) ;

   return ( 1 > i ? 1 :
	    ( m_nref < (unsigned int) i ? m_nref : (unsigned int) i ) ) ;
}

EEDetId 
EcalEndcapGeometry::gId( float x, 
			 float y, 
			 float z ) const
{
   const CCGFloat     fac ( fabs( ( 0 > z ? zeN : zeP )/z ) ) ;
   const unsigned int ix  ( xindex( x*fac, z ) ) ; 
   const unsigned int iy  ( yindex( y*fac, z ) ) ; 
   const unsigned int iz  ( z>0 ? 1 : -1 ) ;

   if( EEDetId::validDetId( ix, iy, iz ) ) 
   {
      return EEDetId( ix, iy, iz ) ; // first try is on target
   }
   else // try nearby coordinates, spiraling out from center
   {
      for( unsigned int i ( 1 ) ; i != 6 ; ++i )
      {
	 for( unsigned int k ( 0 ) ; k != 8 ; ++k )
	 {
	    const int jx ( 0 == k || 4 == k || 5 == k ? +i :
			   ( 1 == k || 5 < k ? -i : 0 ) ) ;
	    const int jy ( 2 == k || 4 == k || 6 == k ? +i :
			   ( 3 == k || 5 == k || 7 == k ? -i : 0 ) ) ;
	    if( EEDetId::validDetId( ix + jx, iy + jy, iz ) ) 
	    {
	       return EEDetId( ix + jx, iy + jy, iz ) ;
	    }
	 }
      }
   }
   return EEDetId() ; // nowhere near any crystal
}


// Get closest cell, etc...
DetId 
EcalEndcapGeometry::getClosestCell( const GlobalPoint& r ) const 
{
   try
   {
      EEDetId mycellID ( gId( r.x(), r.y(), r.z() ) ) ; // educated guess

      if( EEDetId::validDetId( mycellID.ix(), 
			       mycellID.iy(),
			       mycellID.zside() ) )
      {
	 // now get points in convenient ordering

	 Pt3D A;
	 Pt3D B;
	 Pt3D C;
	 Pt3D point(r.x(),r.y(),r.z());
	 // D.K. : equation of plane : AA*x+BB*y+CC*z+DD=0;
	 // finding equation for each edge
	 
	 // ================================================================
	 CCGFloat x,y,z;
	 unsigned offset=0;
	 int zsign=1;
	 //================================================================
      
	 // compute the distance of the point with respect of the 4 crystal lateral planes

	 
	 if( nullptr != getGeometry(mycellID) )
	 {
	    const GlobalPoint& myPosition=getGeometry(mycellID)->getPosition();
	 
	    x=myPosition.x();
	    y=myPosition.y();
	    z=myPosition.z();
	 
	    offset=0;
	    // This will disappear when Andre has applied his fix
	 
	    if(z>0)
	    {
	       if(x>0&&y>0)
		  offset=1;
	       else  if(x<0&&y>0)
		  offset=2;
	       else if(x>0&&y<0)
		  offset=0;
	       else if (x<0&&y<0)
		  offset=3;
	       zsign=1;
	    }
	    else
	    {
	       if(x>0&&y>0)
		  offset=3;
	       else if(x<0&&y>0)
		  offset=2;
	       else if(x>0&&y<0)
		  offset=0;
	       else if(x<0&&y<0)
		  offset=1;
	       zsign=-1;
	    }
	    GlobalPoint corners[8];
	    for(unsigned ic=0;ic<4;++ic)
	    {
	       corners[ic]=getGeometry(mycellID)->getCorners()[(unsigned)((zsign*ic+offset)%4)];
	       corners[4+ic]=getGeometry(mycellID)->getCorners()[(unsigned)(4+(zsign*ic+offset)%4)];
	    }

	    CCGFloat SS[4];
	    for (short i=0; i < 4 ; ++i)
	    {
	       A = Pt3D(corners[i%4].x(),corners[i%4].y(),corners[i%4].z());
	       B = Pt3D(corners[(i+1)%4].x(),corners[(i+1)%4].y(),corners[(i+1)%4].z());
	       C = Pt3D(corners[4+(i+1)%4].x(),corners[4+(i+1)%4].y(),corners[4+(i+1)%4].z());
	       Pl3D plane(A,B,C);
	       plane.normalize();
	       CCGFloat distance = plane.distance(point);
	       if (corners[0].z()<0.) distance=-distance;
	       SS[i] = distance;
	    }
	 
	    // Only one move in necessary direction
	 
	    const bool yout ( 0 > SS[0]*SS[2] ) ;
	    const bool xout ( 0 > SS[1]*SS[3] ) ;
	 
	    if( yout || xout )
	    {
	       const int ydel ( !yout ? 0 :  ( 0 < SS[0] ? -1 : 1 ) ) ;
	       const int xdel ( !xout ? 0 :  ( 0 < SS[1] ? -1 : 1 ) ) ;
	       const unsigned int ix ( mycellID.ix() + xdel ) ;
	       const unsigned int iy ( mycellID.iy() + ydel ) ;
	       const unsigned int iz ( mycellID.zside()     ) ;
	       if( EEDetId::validDetId( ix, iy, iz ) ) 
		  mycellID = EEDetId( ix, iy, iz ) ;
	    }
  
	    return mycellID;
	 }
	 return DetId(0);
      }
   }
   catch ( cms::Exception &e ) 
   { 
      return DetId(0);
   }
   return DetId(0);
}

CaloSubdetectorGeometry::DetIdSet 
EcalEndcapGeometry::getCells( const GlobalPoint& r, 
			      double             dR ) const 
{
   CaloSubdetectorGeometry::DetIdSet dis ; // return object
   if( 0.000001 < dR )
   {
     if( dR > M_PI/2. ) // this code assumes small dR
      {
	 dis = CaloSubdetectorGeometry::getCells( r, dR ) ; // base class version
      }
      else
      {
	 const float dR2  ( dR*dR ) ;
	 const float reta ( r.eta() ) ;
	 const float rphi ( r.phi() ) ;
	 const float rx   ( r.x() ) ;
	 const float ry   ( r.y() ) ;
	 const float rz   ( r.z() ) ;
	 const float fac  ( std::abs( zeP/rz ) ) ;
	 const float xx   ( rx*fac ) ; // xyz at endcap z
	 const float yy   ( ry*fac ) ; 
	 const float zz   ( rz*fac ) ; 

	 const float xang  ( std::atan( xx/zz ) ) ;
	 const float lowX  ( zz>0 ? zz*std::tan( xang - dR ) : zz*std::tan( xang + dR ) ) ;
	 const float highX ( zz>0 ? zz*std::tan( xang + dR ) : zz*std::tan( xang - dR ) ) ;
	 const float yang  ( std::atan( yy/zz ) ) ;
	 const float lowY  ( zz>0 ? zz*std::tan( yang - dR ) : zz*std::tan( yang + dR ) ) ;
	 const float highY ( zz>0 ? zz*std::tan( yang + dR ) : zz*std::tan( yang - dR ) ) ;

	 const float refxlo ( 0 > rz ? m_xlo[0] : m_xlo[1] ) ;
	 const float refxhi ( 0 > rz ? m_xhi[0] : m_xhi[1] ) ;
	 const float refylo ( 0 > rz ? m_ylo[0] : m_ylo[1] ) ;
	 const float refyhi ( 0 > rz ? m_yhi[0] : m_yhi[1] ) ;

	 if( lowX  <  refxhi &&   // proceed if any possible overlap with the endcap
	     lowY  <  refyhi &&
	     highX >  refxlo &&
	     highY >  refylo    )
	 {
	    const int ix_ctr ( xindex( xx, rz ) ) ;
	    const int iy_ctr ( yindex( yy, rz ) ) ;
	    const int iz     ( rz>0 ? 1 : -1 ) ;
	    
	    const int ix_hi  ( ix_ctr + int( ( highX - xx )/m_wref ) + 2 ) ;
	    const int ix_lo  ( ix_ctr - int( ( xx - lowX  )/m_wref ) - 2 ) ;
	    
	    const int iy_hi  ( iy_ctr + int( ( highY - yy )/m_wref ) + 2 ) ;
	    const int iy_lo  ( iy_ctr - int( ( yy - lowY  )/m_wref ) - 2 ) ;
	    
	    for( int kx ( ix_lo ) ; kx <= ix_hi ; ++kx ) 
	    {
	       if( kx >  0      && 
		   kx <= (int) m_nref    )
	       {
		  for( int ky ( iy_lo ) ; ky <= iy_hi ; ++ky ) 
		  {
		     if( ky >  0      && 
			 ky <= (int) m_nref    )
		     {
			if( EEDetId::validDetId( kx, ky, iz ) ) // reject invalid ids
			{
			  const EEDetId id ( kx, ky, iz ) ;
			  const CaloCellGeometry* cell  = &m_cellVec[ id.denseIndex()];
			  const float       eta  (cell->etaPos() ) ;
			  const float       phi  (cell->phiPos() ) ;
			  if( reco::deltaR2( eta, phi, reta, rphi ) < dR2 ) dis.insert( id ) ;
			}
		     }
		  }
	       }
	    }
	 }
      }
   }
   return dis;
}

const EcalEndcapGeometry::OrderedListOfEBDetId*
EcalEndcapGeometry::getClosestBarrelCells( EEDetId id ) const
{
   OrderedListOfEBDetId* ptr ( nullptr ) ;
   auto ptrVec = m_borderPtrVec.load(std::memory_order_acquire);
   if(!ptrVec) {
       if(0 != id.rawId() && nullptr != getGeometry(id)) {
           const float phi(370.+getGeometry(id)->getPosition().phi().degrees());
           const int iPhi(1+int(phi)%360) ;
           const int iz(id.zside()) ;
           if (!m_borderMgr.load(std::memory_order_acquire)) {
               EZMgrFL<EBDetId>* expect = nullptr;
               auto ptrMgr = new EZMgrFL<EBDetId>( 720*9, 9 );
               bool exchanged = m_borderMgr.compare_exchange_strong(expect, ptrMgr, std::memory_order_acq_rel);
               if(!exchanged) delete ptrMgr;
           }
           VecOrdListEBDetIdPtr* expect = nullptr;
           auto ptrVec = new VecOrdListEBDetIdPtr();
           ptrVec->reserve(720);
           for( unsigned int i ( 0 ) ; i != 720 ; ++i )
           {
               const int kz     ( 360>i ? -1 : 1 ) ;
               const int iEta   ( kz*85 ) ;
               const int iEtam1 ( kz*84 ) ;
               const int iEtam2 ( kz*83 ) ;
               const int jPhi   ( i%360 + 1 ) ;
               OrderedListOfEBDetId& olist ( *new OrderedListOfEBDetId( m_borderMgr.load(std::memory_order_acquire) ) );
               olist[0]=EBDetId( iEta  ,        jPhi     ) ;
               olist[1]=EBDetId( iEta  , myPhi( jPhi+1 ) ) ;
               olist[2]=EBDetId( iEta  , myPhi( jPhi-1 ) ) ;
               olist[3]=EBDetId( iEtam1,        jPhi     ) ;
               olist[4]=EBDetId( iEtam1, myPhi( jPhi+1 ) ) ;
               olist[5]=EBDetId( iEtam1, myPhi( jPhi-1 ) ) ;
               olist[6]=EBDetId( iEta  , myPhi( jPhi+2 ) ) ;
               olist[7]=EBDetId( iEta  , myPhi( jPhi-2 ) ) ;
               olist[8]=EBDetId( iEtam2,        jPhi     ) ;
               ptrVec->emplace_back( &olist ) ;
           }
           bool exchanged = m_borderPtrVec.compare_exchange_strong(expect, ptrVec, std::memory_order_acq_rel);
           if(!exchanged) delete ptrVec;
           ptrVec = m_borderPtrVec.load(std::memory_order_acquire);
           ptr = (*ptrVec)[ ( iPhi - 1 ) + ( 0>iz ? 0 : 360 ) ] ;
       }
   }
   return ptr;
}

void
EcalEndcapGeometry::localCorners( Pt3DVec&        lc  ,
				  const CCGFloat* pv  ,
				  unsigned int   /*i*/,
				  Pt3D&           ref   )
{
   TruncatedPyramid::localCorners( lc, pv, ref ) ;
}

void
EcalEndcapGeometry::newCell( const GlobalPoint& f1 ,
			     const GlobalPoint& f2 ,
			     const GlobalPoint& f3 ,
			     const CCGFloat*    parm ,
			     const DetId&       detId   ) 
{
   const unsigned int cellIndex ( EEDetId( detId ).denseIndex() ) ;
   m_cellVec[ cellIndex ] =
      TruncatedPyramid( cornersMgr(), f1, f2, f3, parm ) ;
   addValidID( detId ) ;
}


CCGFloat 
EcalEndcapGeometry::avgAbsZFrontFaceCenter() const
{
   if(!m_check.load(std::memory_order_acquire))
   {
      CCGFloat sum ( 0 ) ;
      for( unsigned int i ( 0 ) ; i != m_cellVec.size() ; ++i )
      {
	 const CaloCellGeometry* cell ( cellGeomPtr(i) ) ;
	 if( nullptr != cell )
	 {
	    sum += fabs( cell->getPosition().z() ) ;
	 }
      }
      m_avgZ = sum/m_cellVec.size();
      m_check.store(true, std::memory_order_release);
   }
   return m_avgZ;
}

const CaloCellGeometry* 
EcalEndcapGeometry::cellGeomPtr( uint32_t index ) const
{
   const CaloCellGeometry* cell ( &m_cellVec[ index ] ) ;
   return ( m_cellVec.size() < index ||
	    nullptr == cell->param() ? nullptr : cell ) ;
}
