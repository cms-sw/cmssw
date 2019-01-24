#include "DataFormats/ForwardDetId/interface/BTLDetId.h"

/** Returns BTL iphi index for crystal according to type tile or bar */
int BTLDetId::iphi( CrysLayout lay ) const {
  int kCrystalsInPhi = 1;
  switch ( lay ) {
    case CrysLayout::tile : {
      kCrystalsInPhi = kCrystalsInPhiTile ;
      break ;
    }
    case CrysLayout::bar  : {
      kCrystalsInPhi = kCrystalsInPhiBar ;
      break ;
    }
    case CrysLayout::barzflat  : {
      kCrystalsInPhi = kCrystalsInPhiBarZ ;
      break ;
    }
    default: {
      break ;
    }
  }
  return kCrystalsInPhi * ( ( mtdRR()%HALF_ROD > 0 ? mtdRR()%HALF_ROD : HALF_ROD ) -1 ) 
    + ( crystal()%kCrystalsInPhi > 0 ? crystal()%kCrystalsInPhi : kCrystalsInPhi ) ;
}

/** Returns BTL ieta index for crystal according to type tile or bar */
int BTLDetId::ietaAbs( CrysLayout lay ) const {
  int kCrystalsInEta = 1, kCrystalsInPhi = 1;
  std::array<int,4> kTypeBoundaries = {};
  switch ( lay ) {
    case CrysLayout::tile : {
      kCrystalsInEta = kCrystalsInEtaTile;
      kCrystalsInPhi = kCrystalsInPhiTile;
      kTypeBoundaries = kTypeBoundariesReference;
      break ;
    }
    case CrysLayout::bar  : {
      kCrystalsInEta = kCrystalsInEtaBar ;
      kCrystalsInPhi = kCrystalsInPhiBar ;
      kTypeBoundaries = kTypeBoundariesReference;
      break ;
    }
    case CrysLayout::barzflat  : {
      kCrystalsInEta = kCrystalsInEtaBarZ ;
      kCrystalsInPhi = kCrystalsInPhiBarZ ;
      kTypeBoundaries = kTypeBoundariesBarZflat ;
      break ;
    }
    default: {
      break ;
    }
  }
  return kCrystalsInEta * ( module() -1 ) 
    + kCrystalsInEta * kTypeBoundaries[(modType() -1)] 
    + ( (crystal()-1)/kCrystalsInPhi + 1 ) ; 
}

int BTLDetId::hashedIndex( CrysLayout lay ) const { 
  int max_iphi = 1, max_ieta = 1;
  switch ( lay ) {
    case CrysLayout::tile : {
      max_iphi = MAX_IPHI_TILE;
      max_ieta = MAX_IETA_TILE;
      break ;
    }
    case CrysLayout::bar : {
      max_iphi = MAX_IPHI_BAR;
      max_ieta = MAX_IETA_BAR;
      break ;
    }
    case CrysLayout::barzflat : {
      max_iphi = MAX_IPHI_BARZFLAT;
      max_ieta = MAX_IETA_BARZFLAT;
      break ;
    }
    default: {
      break ;
    }
  }
  return (max_ieta + ( zside() > 0 ? ietaAbs( lay ) - 1 : -ietaAbs( lay ) ) )*max_iphi+ iphi( lay ) - 1;
}

/** get a DetId from a compact index for arrays */

BTLDetId BTLDetId::getUnhashedIndex( int hi, CrysLayout lay ) const {
  int max_iphi =1 ,max_ieta = 1, nphi = 0, keta = 0, tmphi = hi + 1;
  std::array<int,4> kTypeBoundaries = {};
  switch ( lay ) {
    case CrysLayout::tile : {
      max_iphi = MAX_IPHI_TILE;
      max_ieta = MAX_IETA_TILE;
      nphi = kCrystalsInPhiTile;
      keta = kCrystalsInEtaTile;
      kTypeBoundaries = kTypeBoundariesReference;
      break ;
    }
    case CrysLayout::bar : {
      max_iphi = MAX_IPHI_BAR;
      max_ieta = MAX_IETA_BAR;
      nphi = kCrystalsInPhiBar;
      keta = kCrystalsInEtaBar;
      kTypeBoundaries = kTypeBoundariesReference;
      break ;
    }
    case CrysLayout::barzflat : {
      max_iphi = MAX_IPHI_BARZFLAT;
      max_ieta = MAX_IETA_BARZFLAT;
      nphi = kCrystalsInPhiBarZ;
      keta = kCrystalsInEtaBarZ;
      kTypeBoundaries = kTypeBoundariesBarZflat;
      break ;
    }
    default: {
      break ;
    }
  }
  int zside = 0, rod = 0, module = 0, modtype = 1, crystal = 0;
  if ( tmphi > max_ieta*max_iphi ) { zside = 1; }
  int ip = (tmphi-1)%max_iphi+1;
  int ie = (tmphi-1)/max_iphi - max_ieta;
  ie = ( zside == 1 ? ie + 1 : -ie ) ; 
  rod = (ip-1)/nphi+1;
  module = (ie-1)/keta+1 ; 
  if ( module > kTypeBoundaries[1] ) { modtype = (module > kTypeBoundaries[2] ? 3 : 2 ) ;  }
  if ( modtype > 1 ) { module = module - kTypeBoundaries[modtype-1]; }
  crystal = ((ip-1)%nphi+1)+((ie-1)%keta)*nphi;
  return  BTLDetId( zside, rod, module, modtype, crystal);
}

BTLDetId BTLDetId::geographicalId( CrysLayout lay ) const {
  // reorganize the modules to count from 0 to 54
  //    (0 to 42 in the case of BarZflat geometry)
  // remove module type
  // remove crystal index

  int boundRef = ( lay == CrysLayout::barzflat ?
		   kTypeBoundariesBarZflat[1]  :
		   kTypeBoundariesReference[1] );

  return BTLDetId(mtdSide(),mtdRR(),module()+boundRef*(modType()-1),0,1);
}

#include <iomanip>

std::ostream& operator<< ( std::ostream& os, const BTLDetId& id ) {
  os << ( MTDDetId& ) id;
  os << " BTL " << std::endl
     << " Side        : " << id.mtdSide() << std::endl
     << " Rod         : " << id.mtdRR() << std::endl
     << " Module      : " << id.module() << std::endl
     << " Crystal type: " << id.modType() << std::endl
     << " Crystal     : " << id.crystal() << std::endl;
  return os;
}
