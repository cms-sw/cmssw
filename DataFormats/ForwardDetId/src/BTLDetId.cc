#include "DataFormats/ForwardDetId/interface/BTLDetId.h"

/** Returns BTL iphi index for crystal according to type tile or bar */
int BTLDetId::iphi( kCrysLayout lay ) const {
  int res = 2*MAX_HASH; 
  int kCrystalsInPhi = 1;
  switch ( lay ) {
  case tile : kCrystalsInPhi = kCrystalsInPhiTile ; break ;
  case bar  : kCrystalsInPhi = kCrystalsInPhiBar ; break ;
  default: break ;  
  }
  return res = kCrystalsInPhi * ( ( mtdRR()%HALF_ROD > 0 ? mtdRR()%HALF_ROD : HALF_ROD ) -1 ) 
    + ( crystal()%kCrystalsInPhi > 0 ? crystal()%kCrystalsInPhi : kCrystalsInPhi ) ;
}

/** Returns BTL ieta index for crystal according to type tile or bar */
int BTLDetId::ietaAbs( kCrysLayout lay ) const {
  int res = 2*MAX_HASH;
  int kCrystalsInEta = 1, kCrystalsInPhi = 1;
  switch ( lay ) {
  case tile : kCrystalsInEta = kCrystalsInEtaTile; kCrystalsInPhi = kCrystalsInPhiTile; break ;
  case bar  : kCrystalsInEta = kCrystalsInEtaBar ; kCrystalsInPhi = kCrystalsInPhiBar ; break ;
  default: break ;  
  }
  return res = kCrystalsInEta * ( module() -1 ) 
    + kCrystalsInEta * kTypeBoundaries[(modType() -1)] 
    + ( (crystal()-1)/kCrystalsInPhi + 1 ) ; 
}

int BTLDetId::hashedIndex( kCrysLayout lay ) const { 
  int max_iphi = 1, max_ieta = 1;
  switch ( lay ) {
  case tile : max_iphi = MAX_IPHI_TILE; max_ieta = MAX_IETA_TILE; break ;
  case bar : max_iphi = MAX_IPHI_BAR; max_ieta = MAX_IETA_BAR; break ;
  default: break ;  
  }
  return (max_ieta + ( zside() > 0 ? ietaAbs( lay ) - 1 : -ietaAbs( lay ) ) )*max_iphi+ iphi( lay ) - 1;
}

/** get a DetId from a compact index for arrays */
/* void BTLDetId::getUnhashedIndex( int hi, BTLDetId id, kCrysLayout lay ) const { */
/*   int max_iphi,max_ieta; */
/*   switch ( lay ) { */
/*   case tile : max_iphi = MAX_IPHI_TILE; max_ieta = MAX_IETA_TILE; break ; */
/*   case bar : max_iphi = MAX_IPHI_BAR; max_ieta = MAX_IETA_BAR; break ; */
/*   default: break ;   */
/*   } */
/* } */

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
