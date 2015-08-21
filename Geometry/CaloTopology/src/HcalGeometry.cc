#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>

#include <Math/Transform3D.h>
#include <Math/EulerAngles.h>

typedef CaloCellGeometry::CCGFloat CCGFloat ;
typedef CaloCellGeometry::Pt3D     Pt3D     ;
typedef CaloCellGeometry::Pt3DVec  Pt3DVec  ;
typedef CaloCellGeometry::Tr3D     Tr3D     ;

HcalGeometry::HcalGeometry(const HcalTopology& topology) :
  theTopology( topology ) {
  init();
}
  
HcalGeometry::~HcalGeometry() {}

void HcalGeometry::init() {
  edm::LogInfo("HcalGeometry") << "HcalGeometry::init() "
			       << " HBSize " << theTopology.getHBSize() 
			       << " HESize " << theTopology.getHESize() 
			       << " HOSize " << theTopology.getHOSize() 
			       << " HFSize " << theTopology.getHFSize();
    
  m_hbCellVec = HBCellVec( theTopology.getHBSize() ) ;
  m_heCellVec = HECellVec( theTopology.getHESize() ) ;
  m_hoCellVec = HOCellVec( theTopology.getHOSize() ) ;
  m_hfCellVec = HFCellVec( theTopology.getHFSize() ) ;
}

void HcalGeometry::fillDetIds() const {
  if(!m_emptyIds.isSet()) {
    std::unique_ptr<std::vector<DetId>> p_hbIds{new std::vector<DetId>};
    std::unique_ptr<std::vector<DetId>> p_heIds{new std::vector<DetId>};
    std::unique_ptr<std::vector<DetId>> p_hoIds{new std::vector<DetId>};
    std::unique_ptr<std::vector<DetId>> p_hfIds{new std::vector<DetId>};
    std::unique_ptr<std::vector<DetId>> p_emptyIds{new std::vector<DetId>};
    
    const std::vector<DetId>& baseIds (CaloSubdetectorGeometry::getValidDetIds());
    for (unsigned int i ( 0 ) ; i != baseIds.size() ; ++i) {
      const DetId id ( baseIds[i] );
      if (id.subdetId() == HcalBarrel) {
	p_hbIds->push_back( id ) ;
      } else if (id.subdetId() == HcalEndcap) {
	p_heIds->push_back( id ) ;
      } else if (id.subdetId() == HcalOuter)  {
	p_hoIds->push_back( id ) ;
      } else if (id.subdetId() == HcalForward) {
	p_hfIds->push_back( id ) ;
      }
    }
    std::sort( p_hbIds->begin(), p_hbIds->end() ) ;
    std::sort( p_heIds->begin(), p_heIds->end() ) ;
    std::sort( p_hoIds->begin(), p_hoIds->end() ) ;
    std::sort( p_hfIds->begin(), p_hfIds->end() ) ;
    p_emptyIds->resize( 0 ) ;

    m_hbIds.set(std::move(p_hbIds));
    m_heIds.set(std::move(p_heIds));
    m_hoIds.set(std::move(p_hoIds));
    m_hfIds.set(std::move(p_hfIds));
    m_emptyIds.set(std::move(p_emptyIds));
  }
}

const std::vector<DetId>& 
HcalGeometry::getValidDetIds( DetId::Detector det,
			      int             subdet ) const {
  if( 0 != subdet && !m_hbIds.isSet() ) fillDetIds() ;
  return ( 0 == subdet ? CaloSubdetectorGeometry::getValidDetIds() :
	   ( HcalBarrel == subdet ? *m_hbIds.load() :
	     ( HcalEndcap == subdet ? *m_heIds.load() :
	       ( HcalOuter == subdet ? *m_hoIds.load() :
		 ( HcalForward == subdet ? *m_hfIds.load() : *m_emptyIds.load() ) ) ) ) ) ;
}

DetId HcalGeometry::getClosestCell(const GlobalPoint& r) const {

  // Now find the closest eta_bin, eta value of a bin i is average
  // of eta[i] and eta[i-1]
  static const double z_long=1100.0;
  double abseta = fabs(r.eta());
  double absz   = fabs(r.z());
  
  // figure out subdetector, giving preference to HE in HE/HF overlap region
  HcalSubdetector bc= HcalEmpty;
  if (abseta <= theTopology.etaMax(HcalBarrel) ) {
    bc = HcalBarrel;
  } else if (absz >= z_long) {
    bc = HcalForward;
  } else if (theTopology.etaMax(HcalEndcap) ) {
    bc = HcalEndcap;
  } else {
    bc = HcalForward;
  }

  // find eta bin
  int etaring = etaRing(bc, abseta);

  int phibin = phiBin(bc, etaring, r.phi());

  // add a sign to the etaring
  int etabin = (r.z() > 0) ? etaring : -etaring;

  if (bc == HcalForward) {
    static const double z_short=1137.0;
    // Next line is premature depth 1 and 2 can coexist for large z-extent
    //    HcalDetId bestId(bc,etabin,phibin,((fabs(r.z())>=z_short)?(2):(1)));
    // above line is no good with finite precision
    HcalDetId bestId(bc,etabin,phibin,((fabs(r.z()) - z_short >-0.1)?(2):(1)));
    return bestId;
  } else {

    //Now do depth if required
    int dbin = 1;
    double pointrz=0, drz=99999.;
    HcalDetId currentId(bc, etabin, phibin, dbin);
    if (bc == HcalBarrel) pointrz = r.mag();
    else                  pointrz = std::abs(r.z());
    HcalDetId bestId;
    for ( ; currentId != HcalDetId(); theTopology.incrementDepth(currentId)) {
      const CaloCellGeometry * cell = getGeometry(currentId);
      if (cell == 0) {
        assert (bestId != HcalDetId());
        break;
      } else {
        double rz;
        if (bc == HcalEndcap) rz = std::abs(cell->getPosition().z());
        else                  rz = cell->getPosition().mag();
        if (std::abs(pointrz-rz)<drz) {
          bestId = currentId;
          drz    = std::abs(pointrz-rz);
        }
      }
    }
    
    return bestId;
  }
}


int HcalGeometry::etaRing(HcalSubdetector bc, double abseta) const {
  return theTopology.etaRing(bc, abseta);
}

int HcalGeometry::phiBin(HcalSubdetector bc, int etaring, double phi) const {
  return theTopology.phiBin(bc, etaring, phi);
}

CaloSubdetectorGeometry::DetIdSet HcalGeometry::getCells(const GlobalPoint& r, 
							 double dR ) const {
  CaloSubdetectorGeometry::DetIdSet dis;  // this is the return object

  if (0.000001 < dR) {
    if (dR > M_PI/2.) {// this version needs "small" dR
      dis = CaloSubdetectorGeometry::getCells(r, dR); // base class version
    } else {
      const double dR2     ( dR*dR ) ;
      const double reta    ( r.eta() ) ;
      const double rphi    ( r.phi() ) ;
      const double lowEta  ( reta - dR ) ;
      const double highEta ( reta + dR ) ;
      const double lowPhi  ( rphi - dR ) ;
      const double highPhi ( rphi + dR ) ;
       
      const double hfEtaHi (theTopology.etaMax(HcalForward));
	 
      if (highEta > -hfEtaHi &&
	  lowEta  <  hfEtaHi    ) { // in hcal
	const HcalSubdetector hs[] = {HcalBarrel, HcalOuter, HcalEndcap, HcalForward } ;

	for (unsigned int is ( 0 ) ; is != 4 ; ++is ) {
	  const int sign        (  reta>0 ? 1 : -1 ) ;
	  const int ieta_center ( sign*etaRing( hs[is], fabs( reta ) ) ) ;
	  const int ieta_lo     ( ( 0 < lowEta*sign ? sign : -sign )*etaRing( hs[is], fabs( lowEta ) ) ) ;
	  const int ieta_hi     ( ( 0 < highEta*sign ? sign : -sign )*etaRing( hs[is], fabs( highEta ) ) ) ;
          const int iphi_lo     ( phiBin( hs[is], ieta_center, lowPhi  ) ) ;
          const int iphi_hi     ( phiBin( hs[is], ieta_center, highPhi ) ) ;
	  const int jphi_lo     ( iphi_lo>iphi_hi ? iphi_lo - 72 : iphi_lo ) ;
	  const int jphi_hi     ( iphi_hi ) ;

	   const int idep_lo     ( 1 == is ? 4 : 1 ) ;
	   const int idep_hi     ( theTopology.maxDepth(hs[is]) );
	   for (int ieta ( ieta_lo ) ; ieta <= ieta_hi ; ++ieta) {// over eta limits
	     if (ieta != 0) {
	       for (int jphi ( jphi_lo ) ; jphi <= jphi_hi ; ++jphi) { // over phi limits
		 const int iphi ( 1 > jphi ? jphi+72 : jphi ) ;
		 for (int idep ( idep_lo ) ; idep <= idep_hi ; ++idep ) {
		   const HcalDetId did ( hs[is], ieta, iphi, idep ) ;
		   if (theTopology.valid(did)) {
		     const CaloCellGeometry* cell ( getGeometry( did ) );
		     if (0 != cell ) {
		       const GlobalPoint& p   ( cell->getPosition() ) ;
		       const double       eta ( p.eta() ) ;
		       const double       phi ( p.phi() ) ;
		       if (reco::deltaR2(eta, phi, reta, rphi ) < dR2) dis.insert( did ) ;
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



DetId HcalGeometry::detIdFromBarrelAlignmentIndex(unsigned int i) {
   assert( i < numberOfBarrelAlignments() ) ;
   const int ieta  ( i < numberOfBarrelAlignments()/2 ? -1 : 1 ) ;
   const int iphi ( 1 + (4*i)%72 ) ;
   return HcalDetId( HcalBarrel, ieta, iphi, 1 ) ;
}

DetId HcalGeometry::detIdFromEndcapAlignmentIndex(unsigned int i) {
   assert( i < numberOfEndcapAlignments() ) ;
   const int ieta  ( i < numberOfEndcapAlignments()/2 ? -16 : 16 ) ;
   const int iphi ( 1 + (4*i)%72 ) ;
   return HcalDetId( HcalEndcap, ieta, iphi, 1 ) ;
}

DetId HcalGeometry::detIdFromForwardAlignmentIndex(unsigned int i) {
   assert( i < numberOfForwardAlignments() ) ;
   const int ieta ( i < numberOfForwardAlignments()/2 ? -29 : 29 ) ;
   const int iphi ( 1 + (4*i)%72 ) ;
   return HcalDetId( HcalForward, ieta, iphi, 1 ) ;
}

DetId HcalGeometry::detIdFromOuterAlignmentIndex(unsigned int i) {
   assert( i < numberOfOuterAlignments() ) ;
   const int ring ( i/12 ) ;
   const int ieta ( 0 == ring ? -11 :
		    1 == ring ? -5  :
		    2 == ring ?  1  :
		    3 == ring ?  5  : 11 ) ;
   const int iphi ( 1 + ( i - ring*12 )*6 ) ;
   return HcalDetId( HcalOuter, ieta, iphi, 4 ) ;
}

DetId HcalGeometry::detIdFromLocalAlignmentIndex(unsigned int i) {
   assert( i < numberOfAlignments() ) ;

   const unsigned int nB ( numberOfBarrelAlignments()  ) ;
   const unsigned int nE ( numberOfEndcapAlignments()  ) ;
   const unsigned int nF ( numberOfForwardAlignments() ) ;
//   const unsigned int nO ( numberOfOuterAlignments()   ) ;

   return (  i < nB       ? detIdFromBarrelAlignmentIndex( i ) :
	     i < nB+nE    ? detIdFromEndcapAlignmentIndex( i - nB ) :
	     i < nB+nE+nF ? detIdFromForwardAlignmentIndex( i - nB - nE ) :
	     detIdFromOuterAlignmentIndex( i - nB - nE - nF ) ) ;
}

unsigned int HcalGeometry::alignmentBarEndForIndexLocal(const DetId& id ,
							unsigned int nD) {
   const HcalDetId hid ( id ) ;
   const unsigned int iphi ( hid.iphi() ) ;
   const int ieta ( hid.ieta() ) ;
   const unsigned int index ( ( 0 < ieta ? nD/2 : 0 ) + ( iphi + 1 )%72/4 ) ;
   assert( index < nD ) ;
   return index ;
}

unsigned int HcalGeometry::alignmentBarrelIndexLocal(const DetId& id) {
  return alignmentBarEndForIndexLocal( id, numberOfBarrelAlignments() ) ;
}

unsigned int HcalGeometry::alignmentEndcapIndexLocal(const DetId& id) {
  return alignmentBarEndForIndexLocal( id, numberOfEndcapAlignments() ) ;
}

unsigned int HcalGeometry::alignmentForwardIndexLocal(const DetId& id) {
   return alignmentBarEndForIndexLocal( id, numberOfForwardAlignments() ) ;
}

unsigned int HcalGeometry::alignmentOuterIndexLocal(const DetId& id) {
   const HcalDetId hid ( id ) ;
   const int ieta ( hid.ieta() ) ;
   const int iphi ( hid.iphi() ) ;
   const int ring ( ieta < -10 ? 0 :
		    ( ieta < -4 ? 1 :
		      ( ieta < 5 ? 2 :
			( ieta < 11 ? 3 : 4 ) ) ) ) ;

   const unsigned int index ( 12*ring + ( iphi - 1 )/6 ) ;
   assert( index < numberOfOuterAlignments() ) ;
   return index ;
}

unsigned int HcalGeometry::alignmentTransformIndexLocal(const DetId& id) {
   assert(id.det() == DetId::Hcal) ;

   const HcalDetId hid ( id ) ;
   bool isHB = (hid.subdet() == HcalBarrel);
   bool isHE = (hid.subdet() == HcalEndcap);
   bool isHF = (hid.subdet() == HcalForward);
   // bool isHO = (hid.subdet() == HcalOuter);

   const unsigned int nB ( numberOfBarrelAlignments()  ) ;
   const unsigned int nE ( numberOfEndcapAlignments()  ) ;
   const unsigned int nF ( numberOfForwardAlignments() ) ;
   // const unsigned int nO ( numberOfOuterAlignments()   ) ;

   const unsigned int index (isHB ? alignmentBarrelIndexLocal(id) :
			     isHE ? alignmentEndcapIndexLocal(id) + nB :
			     isHF ? alignmentForwardIndexLocal( id ) + nB + nE :
			     alignmentOuterIndexLocal(id) + nB + nE + nF
			     );

   assert( index < numberOfAlignments() ) ;
   return index ;
}

unsigned int HcalGeometry::alignmentTransformIndexGlobal( const DetId& id ) {
  return (unsigned int)DetId::Hcal - 1 ;
}

void HcalGeometry::localCorners(Pt3DVec&        lc,
				const CCGFloat* pv,
				unsigned int    i,
				Pt3D&           ref) {
  HcalDetId hid=HcalDetId(theTopology.denseId2detId(i));

   if (hid.subdet() == HcalForward ) {
      IdealZPrism::localCorners( lc, pv, ref ) ;
   } else {
     IdealObliquePrism::localCorners( lc, pv, ref ) ;
   }
}

void HcalGeometry::newCell(const GlobalPoint& f1 ,
			   const GlobalPoint& f2 ,
			   const GlobalPoint& f3 ,
			   const CCGFloat*    parm ,
			   const DetId&       detId) {

  assert (detId.det()==DetId::Hcal);
    
  const HcalDetId hid ( detId ) ;
  unsigned int din=theTopology.detId2denseId(detId);

  edm::LogInfo("HcalGeometry") << " newCell subdet "
 	    << detId.subdetId() << ", raw ID " 
 	    << detId.rawId() << ", hid " << hid << ", din " 
 	    << din << ", index ";
  
  if (hid.subdet()==HcalBarrel) {
    m_hbCellVec[ din ] = IdealObliquePrism( f1, cornersMgr(), parm ) ;
  } else if (hid.subdet()==HcalEndcap) {
    const unsigned int index ( din - m_hbCellVec.size() ) ;
    m_heCellVec[ index ] = IdealObliquePrism( f1, cornersMgr(), parm ) ;
  } else if (hid.subdet()==HcalOuter) {
    const unsigned int index ( din 
			       - m_hbCellVec.size()
			       - m_heCellVec.size() ) ;
    m_hoCellVec[ index ] = IdealObliquePrism( f1, cornersMgr(), parm ) ;
  } else {
    const unsigned int index ( din 
			       - m_hbCellVec.size()
			       - m_heCellVec.size()
			       - m_hoCellVec.size() ) ;
    m_hfCellVec[ index ] = IdealZPrism( f1, cornersMgr(), parm ) ;
  }

  addValidID( detId ) ;
  m_dins.push_back( din );
}

const CaloCellGeometry* HcalGeometry::cellGeomPtr( unsigned int din ) const {
  const CaloCellGeometry* cell ( 0 ) ;
  if( m_hbCellVec.size() > din ) {
    cell = &m_hbCellVec[ din ] ;
  } else {
    if (m_hbCellVec.size() + m_heCellVec.size() > din) {
      const unsigned int index (din - m_hbCellVec.size() ) ;
      cell = &m_heCellVec[ index ] ;
    } else if (m_hbCellVec.size()+m_heCellVec.size()+m_hoCellVec.size() > din) {
      const unsigned int index (din - m_hbCellVec.size() - m_heCellVec.size());
      cell = &m_hoCellVec[ index ] ;
    } else if (m_hbCellVec.size() + m_heCellVec.size() + m_hoCellVec.size() +
	       m_hfCellVec.size() > din) {
      const unsigned int index (din - m_hbCellVec.size() - m_heCellVec.size() -
				m_hoCellVec.size() ) ;
      cell = &m_hfCellVec[ index ] ;
    }
  }
   
  return (( 0 == cell || 0 == cell->param()) ? 0 : cell ) ;
}

void HcalGeometry::getSummary(CaloSubdetectorGeometry::TrVec&  tVec,
			      CaloSubdetectorGeometry::IVec&   iVec,
			      CaloSubdetectorGeometry::DimVec& dVec,
			      CaloSubdetectorGeometry::IVec& dinsVec ) const {
  tVec.reserve(theTopology.ncells()*numberOfTransformParms() ) ;
  iVec.reserve( numberOfShapes()==1 ? 1 : theTopology.ncells() ) ;
  dVec.reserve( numberOfShapes()*numberOfParametersPerShape() ) ;
  dinsVec.reserve(theTopology.ncells());
   
  for (ParVecVec::const_iterator ivv (parVecVec().begin()) ; 
       ivv != parVecVec().end() ; ++ivv) {
    const ParVec& pv ( *ivv ) ;
    for (ParVec::const_iterator iv ( pv.begin() ) ; iv != pv.end() ; ++iv) {
      dVec.push_back( *iv ) ;
    }
  }
   
  for (unsigned int i ( 0 ) ; i < theTopology.ncells() ; ++i) {
    Tr3D tr ;
    const CaloCellGeometry* ptr ( cellGeomPtr( i ) ) ;
       
    if (0 != ptr) {
      dinsVec.push_back( i );

      ptr->getTransform( tr, ( Pt3DVec* ) 0 ) ;

      if( Tr3D() == tr ) { // for preshower there is no rotation
	const GlobalPoint& gp ( ptr->getPosition() ) ; 
	tr = HepGeom::Translate3D( gp.x(), gp.y(), gp.z() ) ;
      }

      const CLHEP::Hep3Vector  tt ( tr.getTranslation() ) ;
      tVec.push_back( tt.x() ) ;
      tVec.push_back( tt.y() ) ;
      tVec.push_back( tt.z() ) ;
      if (6 == numberOfTransformParms()) {
	const CLHEP::HepRotation rr ( tr.getRotation() ) ;
	const ROOT::Math::Transform3D rtr (rr.xx(), rr.xy(), rr.xz(), tt.x(),
					   rr.yx(), rr.yy(), rr.yz(), tt.y(),
					   rr.zx(), rr.zy(), rr.zz(), tt.z());
	ROOT::Math::EulerAngles ea ;
	rtr.GetRotation( ea ) ;
	tVec.push_back( ea.Phi() ) ;
	tVec.push_back( ea.Theta() ) ;
	tVec.push_back( ea.Psi() ) ;
      }

      const CCGFloat* par ( ptr->param() ) ;

      unsigned int ishape ( 9999 ) ;
      for( unsigned int ivv ( 0 ) ; ivv != parVecVec().size() ; ++ivv ) {
	bool ok ( true ) ;
	const CCGFloat* pv ( &(*parVecVec()[ivv].begin() ) ) ;
	for( unsigned int k ( 0 ) ; k != numberOfParametersPerShape() ; ++k ) {
	  ok = ok && ( fabs( par[k] - pv[k] ) < 1.e-6 ) ;
	}
	if( ok ) {
	  ishape = ivv ;
	  break ;
	}
      }
      assert( 9999 != ishape ) ;
      
      const unsigned int nn (( numberOfShapes()==1) ? (unsigned int)1 : m_dins.size() ) ; 
      if( iVec.size() < nn ) iVec.push_back( ishape ) ;
    }
  }
}
