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

EcalBarrelGeometry::EcalBarrelGeometry() :
   _nnxtalEta     ( 85 ) ,
   _nnxtalPhi     ( 360 ) ,
   _PhiBaskets    ( 18 ) ,
   m_borderMgr    ( 0 ),
   m_borderPtrVec ( 0 ) ,
   m_radius       ( -1. )
{
   const int neba[] = {25,45,65,85} ;
   _EtaBaskets = std::vector<int>( &neba[0], &neba[3] ) ;
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
EcalBarrelGeometry::alignmentTransformIndexGlobal( const DetId& id )
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
  double pointz = r.z();
  int zbin=1;
  if(pointz<0)
    zbin=-1;

  // Now find the closest eta
  double pointeta = r.eta();
  //  double eta;
  double deta=999.;
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

	    double eta = getGeometry(EBDetId(zbin*bin,1,EBDetId::ETAPHIMODE))->getPosition().eta();

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
  const double twopi = M_PI+M_PI;

  // 10 degree tilt
  const double tilt=twopi/36.;
  double pointphi = r.phi()+tilt;

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
      
      HepGeom::Point3D<double>   A;
      HepGeom::Point3D<double>   B;
      HepGeom::Point3D<double>   C;
      HepGeom::Point3D<double>   point(r.x(),r.y(),r.z());

      // D.K. : equation of plane : AA*x+BB*y+CC*z+DD=0;
      // finding equation for each edge

      // Since the point can lie between crystals, it is necessary to keep track of the movements
      // to avoid infinite loops
      std::vector<double> history;
      history.resize(4,0.);
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
	  std::vector<double> SS;

	  // compute the distance of the point with respect of the 4 crystal lateral planes
	  for (short i=0; i < 4 ; ++i)
	    {
	      A = HepGeom::Point3D<double> (corners[i%4].x(),corners[i%4].y(),corners[i%4].z());
	      B = HepGeom::Point3D<double> (corners[(i+1)%4].x(),corners[(i+1)%4].y(),corners[(i+1)%4].z());
	      C = HepGeom::Point3D<double> (corners[4+(i+1)%4].x(),corners[4+(i+1)%4].y(),corners[4+(i+1)%4].z());
	      HepGeom::Plane3D<double>  plane(A,B,C);
	      plane.normalize();
	      double distance = plane.distance(point);
	      if(plane.d()>0.) distance=-distance;
	      if (corners[0].z()<0.) distance=-distance;
	      SS.push_back(distance);
	    }

	  // SS's - normals
	  // check position of the point with respect to opposite side of crystal
	  // if SS's have opposite sign, the  point lies inside that crystal

	  if ( ( SS[0]>0.&&SS[2]>0. )||( SS[0]<0.&&SS[2]<0. ) )
	    {
	      levery = 1;
	      if ( history[0]>0. && history[2]>0. && SS[0]<0 && SS[2]<0 &&
		   (abs(SS[0])+abs(SS[2]))> (abs(history[0])+abs(history[2]))) levery = 0  ;
	      if ( history[0]<0. && history[2]<0. && SS[0]>0 && SS[2]>0 &&
		   (abs(SS[0])+abs(SS[2]))> (abs(history[0])+abs(history[2]))) levery = 0  ;


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
		   (abs(SS[1])+abs(SS[3]))> (abs(history[1])+abs(history[3])) )
		{
		  leverx = 0;
		  start = 0;
		}

	      if ( history[1]<0. && history[3]<0. && SS[1]>0 && SS[3]>0 &&
		   (abs(SS[1])+abs(SS[3]))> (abs(history[1])+abs(history[3])) )
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
	  history =SS;
	  
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
	 const double dR2     ( dR*dR ) ;
	 const double reta    ( r.eta() ) ;
	 const double rz      ( r.z()   ) ;
	 const double rphi    ( r.phi() ) ;
	 const double lowEta  ( reta - dR ) ;
	 const double highEta ( reta + dR ) ;
	 
	 if( highEta > -1.5 &&
	     lowEta  <  1.5    ) // in barrel
	 {
	    const double scale       ( maxphi/(2*M_PI) ) ; // angle to index
	    const int    ieta_center ( int( reta*scale + ((rz<0)?(-1):(1))) ) ;
	    const double phi         ( rphi<0 ? rphi + 2*M_PI : rphi ) ;
	    const int    iphi_center ( int( phi*scale + 11. ) ) ; // phi=-9.4deg is iphi=1

	    const double fr    ( dR*scale    ) ; // # crystal widths in dR
	    const double frp   ( 1.08*fr + 1. ) ; // conservatively above fr 
	    const double frm   ( 0.92*fr - 1. ) ; // conservatively below fr
	    const int    idr   ( (int)frp        ) ; // integerize
	    const int    idr2p ( (int)(frp*frp)     ) ;
	    const int    idr2m ( frm > 0 ? int(frm*frm) : 0 ) ;

	    for( int de ( -idr ) ; de <= idr ; ++de ) // over eta limits
	    {
	       int ieta ( de + ieta_center ) ;
	       
	       if( abs(ieta) <= maxeta &&
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
			      const CaloCellGeometry* cell ( getGeometry( id ) );
			      if( 0 != cell )
			      {
				 const GlobalPoint& p   ( cell->getPosition() ) ;
				 const double       eta ( p.eta() ) ;
				 const double       phi ( p.phi() ) ;
				 ok = ( reco::deltaR2( eta, phi, reta, rphi ) < dR2 ) ;
			      }
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

std::vector<HepGeom::Point3D<double> > 
EcalBarrelGeometry::localCorners( const double* pv, 
				  unsigned int  i,
				  HepGeom::Point3D<double> &   ref )
{
   const bool negz ( EBDetId::kSizeForDenseIndexing/2 >  i ) ;
   const bool odd  ( 1 == i%2 ) ;

   return ( ( ( negz  && !odd ) ||
	      ( !negz && odd  )    ) ? TruncatedPyramid::localCornersReflection( pv, ref ) :
	    TruncatedPyramid::localCornersSwap( pv, ref ) ) ;
}

CaloCellGeometry* 
EcalBarrelGeometry::newCell( const GlobalPoint& f1 ,
			     const GlobalPoint& f2 ,
			     const GlobalPoint& f3 ,
			     CaloCellGeometry::CornersMgr* mgr,
			     const double*      parm ,
			     const DetId&       detId   ) 
{
   return ( new TruncatedPyramid( mgr, f1, f2, f3, parm ) ) ;
}

double 
EcalBarrelGeometry::avgRadiusXYFrontFaceCenter() const 
{
   if( 0 > m_radius )
   {
      double sum ( 0 ) ;
      const CaloSubdetectorGeometry::CellCont& cells ( cellGeometries() ) ;
      for( unsigned int i ( 0 ) ; i != cells.size() ; ++i )
      {
	 sum += cells[i]->getPosition().perp() ;
      }
      m_radius = sum/cells.size() ;
   }
   return m_radius ;
}

