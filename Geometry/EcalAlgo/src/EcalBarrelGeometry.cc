#include "Geometry/CaloGeometry/interface/CaloGenericDetId.h"
#include "Geometry/CaloGeometry/interface/TruncatedPyramid.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"

#include <CLHEP/Geometry/Point3D.h>
#include <CLHEP/Geometry/Plane3D.h>
#include <CLHEP/Geometry/Vector3D.h>

#include <iomanip>
#include <iostream>

typedef CaloCellGeometry::CCGFloat CCGFloat ;
typedef CaloCellGeometry::Pt3D     Pt3D     ;
typedef CaloCellGeometry::Pt3DVec  Pt3DVec  ;
typedef HepGeom::Plane3D<CCGFloat> Pl3D     ;

EcalBarrelGeometry::EcalBarrelGeometry() :
   _nnxtalEta     ( 85 ) ,
   _nnxtalPhi     ( 360 ) ,
   _PhiBaskets    ( 18 ) ,
   m_borderMgr    ( 0 ),
   m_borderPtrVec ( 0 ) ,
   m_radius       ( -1. ),
   m_cellVec      ( k_NumberOfCellsForCorners )
{
   const int neba[] = {25,45,65,85} ;
   _EtaBaskets = std::vector<int>( neba, neba+4 ) ;
}


EcalBarrelGeometry::~EcalBarrelGeometry() 
{
   delete m_borderPtrVec ;
   delete m_borderMgr ;
}


unsigned int
EcalBarrelGeometry::alignmentTransformIndexLocal( const DetId& id )
{
   const CaloGenericDetId gid ( id ) ;

   assert( gid.isEB() ) ;

   unsigned int index ( EBDetId(id).ism() - 1 ) ;

   return index ;
}

DetId 
EcalBarrelGeometry::detIdFromLocalAlignmentIndex( unsigned int iLoc )
{
   return EBDetId( iLoc + 1, 1, EBDetId::SMCRYSTALMODE ) ;
}

unsigned int
EcalBarrelGeometry::alignmentTransformIndexGlobal( const DetId& /*id*/ )
{
   return (unsigned int)DetId::Ecal - 1 ;
}
// Get closest cell, etc...
DetId 
EcalBarrelGeometry::getClosestCell(const GlobalPoint& r) const 
{

  // z is the easy one
  int leverx = 1;
  int levery = 1;
  CCGFloat pointz = r.z();
  int zbin=1;
  if(pointz<0)
    zbin=-1;

  // Now find the closest eta
  CCGFloat pointeta = r.eta();
  //  double eta;
  CCGFloat deta=999.;
  int etabin=1;
  
  int guessed_eta = (int)( fabs(pointeta) / 0.0174)+1;
  int guessed_eta_begin = guessed_eta-1;
  int guessed_eta_end   = guessed_eta+1;
  if (guessed_eta_begin < 1) guessed_eta_begin = 1;
  if (guessed_eta_end > 85) guessed_eta_end = 85;

    for(int bin=guessed_eta_begin; bin<= guessed_eta_end; bin++)
      {
	try
	  {
	    if (!present(EBDetId(zbin*bin,1,EBDetId::ETAPHIMODE)))
	      continue;

	    CCGFloat eta = getGeometry(EBDetId(zbin*bin,1,EBDetId::ETAPHIMODE))->etaPos();

	    if(fabs(pointeta-eta)<deta)
	      {
		deta=fabs(pointeta-eta);
		etabin=bin;
	      }
	    else break;
	  }
	catch ( cms::Exception &e ) 
	  {
	  }
      }
    

  // Now the closest phi. always same number of phi bins(!?)
  constexpr CCGFloat twopi = M_PI+M_PI;
  // 10 degree tilt
  constexpr CCGFloat tilt=twopi/36.;

  CCGFloat pointphi = r.phi()+tilt;

  // put phi in correct range (0->2pi)
  if(pointphi > twopi)
    pointphi -= twopi;
  if(pointphi < 0)
    pointphi += twopi;

  //calculate phi bin, distinguish + and - eta
  int phibin = static_cast<int>(pointphi / (twopi/_nnxtalPhi)) + 1;
  //   if(point.z()<0.0)
  //     {
  //       phibin = nxtalPhi/2 - 1 - phibin;
  //       if(phibin<0)
  //         phibin += nxtalPhi;
  //     }
  try
    {
      EBDetId myCell(zbin*etabin,phibin,EBDetId::ETAPHIMODE);

      if (!present(myCell))
	return DetId(0);
      
      Pt3D A;
      Pt3D B;
      Pt3D C;
      Pt3D point(r.x(),r.y(),r.z());

      // D.K. : equation of plane : AA*x+BB*y+CC*z+DD=0;
      // finding equation for each edge

      // Since the point can lie between crystals, it is necessary to keep track of the movements
      // to avoid infinite loops
      CCGFloat history[4]{0.f};

      //
      // stop movement in eta direction when closest cell was found (point between crystals)
      int start = 1;
      int counter = 0;
      // Moving until find closest crystal in eta and phi directions (leverx and levery)
      while  (leverx==1 || levery == 1)
	{
	  leverx = 0;
	  levery = 0;
	  const CaloCellGeometry::CornersVec& corners 
	     ( getGeometry(myCell)->getCorners() ) ;
	  CCGFloat SS[4];

	  // compute the distance of the point with respect of the 4 crystal lateral planes
	  for (short i=0; i < 4 ; ++i)
	    {
	      A = Pt3D(corners[i%4].x(),corners[i%4].y(),corners[i%4].z());
	      B = Pt3D(corners[(i+1)%4].x(),corners[(i+1)%4].y(),corners[(i+1)%4].z());
	      C = Pt3D(corners[4+(i+1)%4].x(),corners[4+(i+1)%4].y(),corners[4+(i+1)%4].z());
	      Pl3D plane(A,B,C);
	      plane.normalize();
	      CCGFloat distance = plane.distance(point);
	      if(plane.d()>0.) distance=-distance;
	      if (corners[0].z()<0.) distance=-distance;
	      SS[i] = distance;
	    }

	  // SS's - normals
	  // check position of the point with respect to opposite side of crystal
	  // if SS's have opposite sign, the  point lies inside that crystal

	  if ( ( SS[0]>0.&&SS[2]>0. )||( SS[0]<0.&&SS[2]<0. ) )
	    {
	      levery = 1;
	      if ( history[0]>0. && history[2]>0. && SS[0]<0 && SS[2]<0 &&
		   (fabs(SS[0])+fabs(SS[2]))> (fabs(history[0])+fabs(history[2]))) levery = 0  ;
	      if ( history[0]<0. && history[2]<0. && SS[0]>0 && SS[2]>0 &&
		   (fabs(SS[0])+fabs(SS[2]))> (fabs(history[0])+fabs(history[2]))) levery = 0  ;


	      if (SS[0]>0. )
		{
		  EBDetId nextPoint;
		  if (myCell.iphi()==EBDetId::MIN_IPHI) 
		    nextPoint=EBDetId(myCell.ieta(),EBDetId::MAX_IPHI);
		  else 
		    nextPoint=EBDetId(myCell.ieta(),myCell.iphi()-1);
		  if (present(nextPoint))
		    myCell=nextPoint;
		  else
		    levery=0;		  
		}
	      else
		{
		  EBDetId nextPoint;
		  if (myCell.iphi()==EBDetId::MAX_IPHI)
		    nextPoint=EBDetId(myCell.ieta(),EBDetId::MIN_IPHI);
		  else
		    nextPoint=EBDetId(myCell.ieta(),myCell.iphi()+1);
		  if (present(nextPoint))
		    myCell=nextPoint;
		  else
		    levery=0;
		}
	    }


	  if ( ( ( SS[1]>0.&&SS[3]>0. )||( SS[1]<0.&&SS[3]<0. )) && start==1  )
	    {
	      leverx = 1;

	      if ( history[1]>0. && history[3]>0. && SS[1]<0 && SS[3]<0 &&
		   (fabs(SS[1])+fabs(SS[3]))> (fabs(history[1])+fabs(history[3])) )
		{
		  leverx = 0;
		  start = 0;
		}

	      if ( history[1]<0. && history[3]<0. && SS[1]>0 && SS[3]>0 &&
		   (fabs(SS[1])+fabs(SS[3]))> (fabs(history[1])+fabs(history[3])) )
		{
		  leverx = 0;
		  start = 0;
		}


	      if (SS[1]>0.)
		{
		  EBDetId nextPoint;
		  if (myCell.ieta()==-1) 
		    nextPoint=EBDetId (1,myCell.iphi());
		  else 
		    {
		      int nieta= myCell.ieta()+1;
		      if(nieta==86) nieta=85;
		      nextPoint=EBDetId(nieta,myCell.iphi());
		    }
		  if (present(nextPoint))
		    myCell = nextPoint;
		  else
		    leverx = 0;
		}
	      else
		{
		  EBDetId nextPoint;
		  if (myCell.ieta()==1) 
		    nextPoint=EBDetId(-1,myCell.iphi());
		  else 
		    {
		      int nieta=myCell.ieta()-1;
		      if(nieta==-86) nieta=-85;
		      nextPoint=EBDetId(nieta,myCell.iphi());
		    }
		  if (present(nextPoint))
		    myCell = nextPoint;
		  else
		    leverx = 0;
		}
	    }
	  
	  // Update the history. If the point lies between crystals, the closest one
	  // is returned
	  std::copy(SS,SS+4,history);
	  
	  counter++;
	  if (counter == 10)
	    {
	      leverx=0;
	      levery=0;
	    }
	}
      // D.K. if point lies netween cells, take a closest cell.
      return DetId(myCell);
    }
  catch ( cms::Exception &e ) 
    { 
      return DetId(0);
    }

}

CaloSubdetectorGeometry::DetIdSet 
EcalBarrelGeometry::getCells( const GlobalPoint& r, 
			      double             dR ) const 
{
   static const int maxphi ( EBDetId::MAX_IPHI ) ;
   static const int maxeta ( EBDetId::MAX_IETA ) ;
   CaloSubdetectorGeometry::DetIdSet dis;  // this is the return object

   if( 0.000001 < dR )
   {
      if( dR > M_PI/2. ) // this version needs "small" dR
      {
	 dis = CaloSubdetectorGeometry::getCells( r, dR ) ; // base class version
      }
      else
      {
	 const float dR2     ( dR*dR ) ;
	 const float reta    ( r.eta() ) ;
	 const float rz      ( r.z()   ) ;
	 const float rphi    ( r.phi() ) ;
	 const float lowEta  ( reta - dR ) ;
	 const float highEta ( reta + dR ) ;
	 
	 if( highEta > -1.5 &&
	     lowEta  <  1.5    ) // in barrel
	 {
	    const float scale       ( maxphi/float(2*M_PI) ) ; // angle to index
	    const int    ieta_center ( int( reta*scale + ((rz<0)?(-1):(1))) ) ;
	    const float phi         ( rphi<0 ? rphi + float(2*M_PI) : rphi ) ;
	    const int    iphi_center ( int( phi*scale + 11.f ) ) ; // phi=-9.4deg is iphi=1

	    const float fr    ( dR*scale    ) ; // # crystal widths in dR
	    const float frp   ( 1.08f*fr + 1.f ) ; // conservatively above fr 
	    const float frm   ( 0.92f*fr - 1.f ) ; // conservatively below fr
	    const int    idr   ( (int)frp        ) ; // integerize
	    const int    idr2p ( (int)(frp*frp)     ) ;
	    const int    idr2m ( frm > 0 ? int(frm*frm) : 0 ) ;

	    for( int de ( -idr ) ; de <= idr ; ++de ) // over eta limits
	    {
	       int ieta ( de + ieta_center ) ;
	       
	       if( std::abs(ieta) <= maxeta &&
		   ieta      != 0         ) // eta is in EB
	       {
		  const int de2 ( de*de ) ;
		  for( int dp ( -idr ) ; dp <= idr ; ++dp )  // over phi limits
		  {
		     const int irange2 ( dp*dp + de2 ) ;
		     
		     if( irange2 <= idr2p ) // cut off corners that must be too far away
		     {
			const int iphi ( ( iphi_center + dp + maxphi - 1 )%maxphi + 1 ) ;
			
			if( iphi != 0 )
			{
			   const EBDetId id ( ieta, iphi ) ;
			   
			   bool ok ( irange2 < idr2m ) ;  // no more calculation necessary if inside this radius
			   
			   if( !ok ) // if not ok, then we have to test this cell for being inside cone
			   {
			     const CaloCellGeometry* cell  = &m_cellVec[ id.denseIndex()];
			     const float       eta ( cell->etaPos() ) ;
			     const float       phi ( cell->phiPos() ) ;
			     ok = ( reco::deltaR2( eta, phi, reta, rphi ) < dR2 ) ;
			   }
			   if( ok ) dis.insert( id ) ;
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

const EcalBarrelGeometry::OrderedListOfEEDetId* 
EcalBarrelGeometry::getClosestEndcapCells( EBDetId id ) const
{
   OrderedListOfEEDetId* ptr ( 0 ) ;
   if( 0 != id.rawId() )
   {
      const int iPhi     ( id.iphi() ) ;

      const int iz       ( id.ieta()>0 ? 1 : -1 ) ;
      const EEDetId eeid ( EEDetId::idOuterRing( iPhi, iz ) ) ;

//      const int ix ( eeid.ix() ) ;
//      const int iy ( eeid.iy() ) ;

      const int iq ( eeid.iquadrant() ) ;
      const int xout ( 1==iq || 4==iq ? 1 : -1 ) ;
      const int yout ( 1==iq || 2==iq ? 1 : -1 ) ;
      if( 0 == m_borderMgr )
      {
	 m_borderMgr = new EZMgrFL<EEDetId>( 720*9, 9 ) ;
      }
      if( 0 == m_borderPtrVec )
      {
	 m_borderPtrVec = new VecOrdListEEDetIdPtr() ;
	 m_borderPtrVec->reserve( 720 ) ;
	 for( unsigned int i ( 0 ) ; i != 720 ; ++i )
	 {
	    const int kz ( 360>i ? -1 : 1 ) ;
	    const EEDetId eeid ( EEDetId::idOuterRing( i%360+1, kz ) ) ;

	    const int jx ( eeid.ix() ) ;
	    const int jy ( eeid.iy() ) ;

	    OrderedListOfEEDetId& olist ( *new OrderedListOfEEDetId( m_borderMgr ) );
	    int il ( 0 ) ;

	    for( unsigned int k ( 1 ) ; k <= 25 ; ++k )
	    {
	       const int kx ( 1==k || 2==k || 3==k || 12==k || 13==k ? 0 :
			      ( 4==k || 6==k || 8==k || 15==k || 20==k ? 1 :
				( 5==k || 7==k || 9==k || 16==k || 19==k ? -1 :
				  ( 10==k || 14==k || 21==k || 22==k || 25==k ? 2 : -2 )))) ;
	       const int ky ( 1==k || 4==k || 5==k || 10==k || 11==k ? 0 :
			      ( 2==k || 6==k || 7==k || 14==k || 17==k ? 1 :
				( 3==k || 8==k || 9==k || 18==k || 21==k ? -1 :
				  ( 12==k || 15==k || 16==k || 22==k || 23==k ? 2 : -2 )))) ;

	       if( 8>=il && EEDetId::validDetId( jx + kx*xout ,
						 jy + ky*yout , kz ) ) 
	       {
		  olist[il++]=EEDetId( jx + kx*xout ,
				       jy + ky*yout , kz ) ;
	       }
	    }
	    m_borderPtrVec->push_back( &olist ) ;
	 }
      }
      ptr = (*m_borderPtrVec)[ iPhi - 1 + ( 0>iz ? 0 : 360 ) ] ;
   }
   return ptr ;
}

void
EcalBarrelGeometry::localCorners( Pt3DVec&        lc  ,
				  const CCGFloat* pv  , 
				  unsigned int    i   ,
				  Pt3D&           ref   )
{
   const bool negz ( EBDetId::kSizeForDenseIndexing/2 >  i ) ;
   const bool odd  ( 1 == i%2 ) ;

   if( ( ( negz  && !odd ) ||
	 ( !negz && odd  )    ) )
   {
      TruncatedPyramid::localCornersReflection( lc, pv, ref ) ;
   }
   else
   {
      TruncatedPyramid::localCornersSwap( lc, pv, ref ) ;
   }
}

void
EcalBarrelGeometry::newCell( const GlobalPoint& f1 ,
			     const GlobalPoint& f2 ,
			     const GlobalPoint& f3 ,
			     const CCGFloat*    parm ,
			     const DetId&       detId   ) 
{
   const unsigned int cellIndex ( EBDetId( detId ).denseIndex() ) ;
   m_cellVec[ cellIndex ] =
      TruncatedPyramid( cornersMgr(), f1, f2, f3, parm ) ;
   m_validIds.push_back( detId ) ;
}

CCGFloat 
EcalBarrelGeometry::avgRadiusXYFrontFaceCenter() const 
{
   if( 0 > m_radius )
   {
      CCGFloat sum ( 0 ) ;
      for( uint32_t i ( 0 ) ; i != m_cellVec.size() ; ++i )
      {
	 const CaloCellGeometry* cell ( cellGeomPtr(i) ) ;
	 if( 0 != cell )
	 {
	    const GlobalPoint& pos ( cell->getPosition() ) ;
	    sum += pos.perp() ;
	 }
      }
      m_radius = sum/m_cellVec.size() ;
   }
   return m_radius ;
}

const CaloCellGeometry* 
EcalBarrelGeometry::cellGeomPtr( uint32_t index ) const
{
   const CaloCellGeometry* cell ( &m_cellVec[ index ] ) ;
   return ( m_cellVec.size() < index ||
	    0 == cell->param() ? 0 : cell ) ;
}
