#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGenericDetId.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/HcalTowerAlgo//src/HcalHardcodeGeometryData.h"
#include <algorithm>

typedef CaloCellGeometry::CCGFloat CCGFloat ;
typedef CaloCellGeometry::Pt3D     Pt3D     ;
typedef CaloCellGeometry::Pt3DVec  Pt3DVec  ;

HcalGeometry::HcalGeometry( const HcalTopology* topology ) :
    theTopology( topology )
{
   init();
}
  

HcalGeometry::~HcalGeometry() 
{}


void
HcalGeometry::init()
{
    // SLHC
    //
    // kHBSizePreLS1 = 2*kHBhalf
    // kHESizePreLS1 = 2*kHEhalf
    // kHBHalfExtra  = 72*(maxDepthHB*15-16)
    // kHEHalfExtra  = 36*(maxDepthHE*19-40)
    // kHBSize = kHBSizePreLS1+kHBSizeExtra
    // kHESize = kHESizePreLS1+kHESizeExtra

    // Current
    //
    // kHBSize = 2*kHBhalf
    // kHESize = 2*kHEhalf
    
    std::cout << "HcalGeometry::init() "
	      << "HcalDetId::kHBSize " << theTopology->getHBSize() << " (HcalDetId::kHBSize " << HcalDetId::kHBSize << " ), "
	      << "HcalDetId::kHESize " << theTopology->getHESize() << " (HcalDetId::kHESize " << HcalDetId::kHESize << " ), "
	      << "HcalDetId::kHOSize " << HcalDetId::kHOSize
	      << "HcalDetId::kHFSize " << HcalDetId::kHFSize << std::endl;
    
    m_hbCellVec = HBCellVec( theTopology->getHBSize() ) ;
    m_heCellVec = HECellVec( theTopology->getHESize() ) ;
   m_hoCellVec = HOCellVec( HcalDetId::kHOSize ) ;
   m_hfCellVec = HFCellVec( HcalDetId::kHFSize ) ;
}

void
HcalGeometry::fillDetIds() const
{
   const std::vector<DetId>& baseIds ( CaloSubdetectorGeometry::getValidDetIds() ) ;
   for( unsigned int i ( 0 ) ; i != baseIds.size() ; ++i ) 
   {
      const DetId id ( baseIds[i] );
      if( id.subdetId() == HcalBarrel )
      { 
	 m_hbIds.push_back( id ) ;
      }
      else
      {
	 if( id.subdetId() == HcalEndcap )
	 { 
	    m_heIds.push_back( id ) ;
	 }
	 else
	 {
	    if( id.subdetId() == HcalOuter )
	    { 
	       m_hoIds.push_back( id ) ;
	    }
	    else
	    {
	       if( id.subdetId() == HcalForward )
	       { 
		  m_hfIds.push_back( id ) ;
	       }
	    }
	 }
      }
   }
   std::sort( m_hbIds.begin(), m_hbIds.end() ) ;   
   std::sort( m_heIds.begin(), m_heIds.end() ) ;   
   std::sort( m_hoIds.begin(), m_hoIds.end() ) ;   
   std::sort( m_hfIds.begin(), m_hfIds.end() ) ;   
   m_emptyIds.resize( 0 ) ;
}

const std::vector<DetId>& 
HcalGeometry::getValidDetIds( DetId::Detector det,
			      int             subdet ) const 
{
   if( 0 != subdet &&
       0 == m_hbIds.size() ) fillDetIds() ;
   return ( 0 == subdet ? CaloSubdetectorGeometry::getValidDetIds() :
	    ( HcalBarrel == subdet ? m_hbIds :
	      ( HcalEndcap == subdet ? m_heIds :
		( HcalOuter == subdet ? m_hoIds :
		  ( HcalForward == subdet ? m_hfIds : m_emptyIds ) ) ) ) ) ;
}

DetId HcalGeometry::getClosestCell(const GlobalPoint& r) const {

  // Now find the closest eta_bin, eta value of a bin i is average
  // of eta[i] and eta[i-1]
  double abseta = fabs(r.eta());
  
  // figure out subdetector, giving preference to HE in HE/HF overlap region
  HcalSubdetector bc= HcalEmpty;
  if (abseta <= theHBHEEtaBounds[theTopology->lastHBRing()] ) {
    bc = HcalBarrel;
  } else if (abseta <= theHBHEEtaBounds[theTopology->lastHERing()] ) {
    bc = HcalEndcap;
  } else {
    bc = HcalForward;
  }

  if (bc == HcalForward) {
    static const double z_short=1137.0;
    int etaring = etaRing(bc, abseta);  // This is safer
    /*
      static const double z_long=1115.0;
      // determine front-face eta
      double radius=sqrt(pow(r.x(),2)+pow(r.y(),2));
      double trueAeta=asinh(z_long/radius);
      // find eta bin
      int etaring = etaRing(bc, trueAeta);
    */
    if (etaring>theTopology->lastHFRing()) etaring=theTopology->lastHFRing(); 
  
    int phibin = phiBin(r.phi(), etaring);

    // add a sign to the etaring
    int etabin = (r.z() > 0) ? etaring : -etaring;
    // Next line is premature depth 1 and 2 can coexist for large z-extent

//    HcalDetId bestId(bc,etabin,phibin,((fabs(r.z())>=z_short)?(2):(1)));
// above line is no good with finite precision
    HcalDetId bestId(bc,etabin,phibin,((fabs(r.z()) - z_short >-0.1)?(2):(1)));
    return bestId;
  } else {

    // find eta bin
    int etaring = etaRing(bc, abseta);
    
    int phibin = phiBin(r.phi(), etaring);
    
    // add a sign to the etaring
    int etabin = (r.z() > 0) ? etaring : -etaring;
    
    //Now do depth if required
    int dbin = 1;
    double pointrz=0, drz=99999.;
    HcalDetId currentId(bc, etabin, phibin, dbin);
    if (bc == HcalBarrel) pointrz = r.mag();
    else                  pointrz = std::abs(r.z());
    HcalDetId bestId;
    for ( ; currentId != HcalDetId(); theTopology->incrementDepth(currentId)) {
      const CaloCellGeometry * cell = getGeometry(currentId);
      assert(cell != 0);
      double rz;
      if (bc == HcalEndcap) rz = std::abs(cell->getPosition().z());
      else                  rz = cell->getPosition().mag();
      if (std::abs(pointrz-rz)<drz) {
	bestId = currentId;
	drz    = std::abs(pointrz-rz);
      }
    }
    
    return bestId;
  }
}


int HcalGeometry::etaRing(HcalSubdetector bc, double abseta) const
{
  int etaring;
  if( bc == HcalForward ) {
    for(etaring = theTopology->firstHFRing();
        etaring <= theTopology->lastHFRing(); ++etaring)
    {
      if(theHFEtaBounds[etaring-theTopology->firstHFRing()+1] > abseta) break;
    }
  }
  else
  {
    for(etaring = 1;
        etaring <= theTopology->lastHERing(); ++etaring)
    {
      if(theHBHEEtaBounds[etaring] >= abseta) break;
    }
  }

  return etaring;
}


int HcalGeometry::phiBin(double phi, int etaring) const
{
   static const double twopi = M_PI+M_PI;
  //put phi in correct range (0->2pi)
  if(phi<0.0) phi += twopi;
  if(phi>twopi) phi -= twopi;
  int nphibins = theTopology->nPhiBins(etaring);
  int phibin= static_cast<int>(phi/twopi*nphibins)+1;
  int iphi;

  // rings 40 and 41 are offset wrt the other phi numbering
  //  1        1         1         2
  //  ------------------------------
  //  72       36        36        1
  if(etaring >= theTopology->firstHFQuadPhiRing())
  {
    phi+=(twopi/36); //shift by half tower.    
    phibin=static_cast<int>(phi/twopi*nphibins);
    if (phibin==0) phibin=18;
    iphi=phibin*4-1; // 71,3,5,
  } else {
    // convert to the convention of numbering 1,3,5, in 36 phi bins
    iphi=(phibin-1)*(72/nphibins) + 1;
  }

  return iphi;
}

CaloSubdetectorGeometry::DetIdSet 
HcalGeometry::getCells( const GlobalPoint& r, 
			double             dR ) const 
{
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
	 const double rphi    ( r.phi() ) ;
	 const double lowEta  ( reta - dR ) ;
	 const double highEta ( reta + dR ) ;
	 const double lowPhi  ( rphi - dR ) ;
	 const double highPhi ( rphi + dR ) ;
	 
	 const double hfEtaHi ( theHFEtaBounds[ theTopology->lastHFRing() -
						theTopology->firstHFRing() + 1 ] ) ;
	 
	 if( highEta > -hfEtaHi &&
	     lowEta  <  hfEtaHi    ) // in hcal
	 {
	    const HcalSubdetector hs[] = { HcalBarrel, HcalOuter, HcalEndcap, HcalForward } ;

	    for( unsigned int is ( 0 ) ; is != 4 ; ++is )
	    {
	       const int sign        (  reta>0 ? 1 : -1 ) ;
	       const int ieta_center ( sign*etaRing( hs[is], fabs( reta ) ) ) ;
	       const int ieta_lo     ( ( 0 < lowEta*sign ? sign : -sign )*etaRing( hs[is], fabs( lowEta ) ) ) ;
	       const int ieta_hi     ( ( 0 < highEta*sign ? sign : -sign )*etaRing( hs[is], fabs( highEta ) ) ) ;
	       const int iphi_lo     ( phiBin( lowPhi , ieta_center ) ) ;
	       const int iphi_hi     ( phiBin( highPhi, ieta_center ) ) ;
	       const int jphi_lo     ( iphi_lo>iphi_hi ? iphi_lo - 72 : iphi_lo ) ;
	       const int jphi_hi     ( iphi_hi ) ;

	       const int idep_lo     ( 1 == is ? 4 : 1 ) ;
	       const int idep_hi     ( 1 == is ? 4 :
				       ( 2 == is ? 3 : 2 ) ) ;
	       for( int ieta ( ieta_lo ) ; ieta <= ieta_hi ; ++ieta ) // over eta limits
	       {
		  if( ieta != 0 )
		  {
		     for( int jphi ( jphi_lo ) ; jphi <= jphi_hi ; ++jphi )  // over phi limits
		     {
			const int iphi ( 1 > jphi ? jphi+72 : jphi ) ;

			for( int idep ( idep_lo ) ; idep <= idep_hi ; ++idep )
			{
			   if( HcalDetId::validDetId( hs[is], ieta, iphi, idep ) )
			   {
			      const HcalDetId did ( hs[is], ieta, iphi, idep ) ;
			      const CaloCellGeometry* cell ( getGeometry( did ) );
			      if( 0 != cell )
			      {
				 const GlobalPoint& p   ( cell->getPosition() ) ;
				 const double       eta ( p.eta() ) ;
				 const double       phi ( p.phi() ) ;
				 if( reco::deltaR2( eta, phi, reta, rphi ) < dR2 ) dis.insert( did ) ;
			      }
			   }
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


unsigned int
HcalGeometry::alignmentTransformIndexLocal( const DetId& id )
{
   const CaloGenericDetId gid ( id ) ;

   assert( gid.isHcal() ) ;


   const HcalDetId hid ( id ) ;

   const int jz ( ( hid.zside() + 1 )/2 ) ;

   const int zoff ( jz*numberOfAlignments()/2 ) ;

   const int detoff ( zoff + 
		      ( gid.isHB() ? 0 :
			( gid.isHE() ? numberOfBarrelAlignments()/2 :
			  ( gid.isHF() ? ( numberOfBarrelAlignments() +
					   numberOfEndcapAlignments() )/2 :
			    ( numberOfBarrelAlignments() +
			      numberOfEndcapAlignments() +
			      numberOfForwardAlignments() )/2 ) ) ) ) ; 

   const int iphi ( hid.iphi() ) ;

   unsigned int index ( numberOfAlignments() ) ;
   if( gid.isHO() )
   {
      const int ieta ( hid.ieta() ) ;
      const int ring ( ieta < -10 ? 0 :
		       ( ieta < -4 ? 1 :
			 ( ieta < 5 ? 2 :
			   ( ieta < 11 ? 3 : 4 ) ) ) ) ;

      index = detoff + 12*ring + ( iphi - 1 )%6 ;
   }
   else
   {
      index = detoff + ( iphi - 1 )%4 ;
   }

   assert( index < numberOfAlignments() ) ;
   return index ;
}

unsigned int
HcalGeometry::alignmentTransformIndexGlobal( const DetId& id )
{
   return (unsigned int)DetId::Hcal - 1 ;
}

void
HcalGeometry::localCorners( Pt3DVec&        lc  ,
			    const CCGFloat* pv  ,
			    unsigned int    i   ,
			    Pt3D&           ref  )
{
   const HcalDetId hid ( HcalDetId::detIdFromDenseIndex( i ) ) ;
   const CaloGenericDetId cgid ( hid ) ;
   if( cgid.isHF() )
   {
      IdealZPrism::localCorners( lc, pv, ref ) ;
   }
   else
   {
      IdealObliquePrism::localCorners( lc, pv, ref ) ;
   }
}

void
HcalGeometry::newCell( const GlobalPoint& f1 ,
		       const GlobalPoint& f2 ,
		       const GlobalPoint& f3 ,
		       const CCGFloat*    parm ,
		       const DetId&       detId   ) 
{
    
   const CaloGenericDetId cgid ( detId ) ;

   const unsigned int din ( cgid.denseIndex() ) ;

//    std::cout << counter++ << ": HcalGeometry::newCell subdet " << detId.subdetId() << ", raw ID " << detId.rawId()
// 	     << ", cgid " << cgid << ", din " << din << std::endl;

   assert( cgid.isHcal() ) ;

   if( cgid.isHB() )
   {
      m_hbCellVec[ din ] = IdealObliquePrism( f1, cornersMgr(), parm ) ;
   }
   else
   {
      if( cgid.isHE() )
      {
	 const unsigned int index ( din - m_hbCellVec.size() ) ;
	 m_heCellVec[ index ] = IdealObliquePrism( f1, cornersMgr(), parm ) ;
      }
      else
      {
	 if( cgid.isHO() )
	 {
	    const unsigned int index ( din 
				       - m_hbCellVec.size()
				       - m_heCellVec.size() ) ;
	    m_hoCellVec[ index ] = IdealObliquePrism( f1, cornersMgr(), parm ) ;
	 }
	 else
	 {
	    const unsigned int index ( din 
				       - m_hbCellVec.size()
				       - m_heCellVec.size()
				       - m_hoCellVec.size() ) ;
	    m_hfCellVec[ index ] = IdealZPrism( f1, cornersMgr(), parm ) ;
	 }
      }
   }
   m_validIds.push_back( detId ) ;
}

const CaloCellGeometry* 
HcalGeometry::cellGeomPtr( uint32_t din ) const
{
   const CaloCellGeometry* cell ( 0 ) ;
   if( m_hbCellVec.size() > din )
   {
      cell = &m_hbCellVec[ din ] ;
   }
   else
   {
      if( m_hbCellVec.size() +
	  m_heCellVec.size() > din )
      {
	 const unsigned int index ( din - m_hbCellVec.size() ) ;
	 cell = &m_heCellVec[ index ] ;
      }
      else
      {
	 if( m_hbCellVec.size() +
	     m_heCellVec.size() +
	     m_hoCellVec.size() > din )
	 {
	    const unsigned int index ( din 
				       - m_hbCellVec.size() 
				       - m_heCellVec.size() ) ;
	    cell = &m_hoCellVec[ index ] ;
	 }
	 else
	 {
	    if( m_hbCellVec.size() +
		m_heCellVec.size() +
		m_hoCellVec.size() +
		m_hfCellVec.size() > din )
	    {
	       const unsigned int index ( din 
					  - m_hbCellVec.size() 
					  - m_heCellVec.size() 
					  - m_hoCellVec.size() ) ;
	       cell = &m_hfCellVec[ index ] ;
	    }
	 }
      }
   }
   return ( 0 == cell || 0 == cell->param() ? 0 : cell ) ;
}
