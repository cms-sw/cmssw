

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"

// default constructor - null DetId
L1CaloRegionDetId::L1CaloRegionDetId() : DetId() { }


// construct from raw id
L1CaloRegionDetId::L1CaloRegionDetId(uint32_t rawid) : DetId(rawid) { }


// construct from ieta, iphi indices
// ieta runs from 0 (at -z) to 21 (at +z)
L1CaloRegionDetId::L1CaloRegionDetId(unsigned ieta, unsigned iphi) :
  DetId(Calo, 2) 
{ 
  id_ |= (ieta & 0x1f) | ((iphi & 0x1f)<<5);
}


// construct from RCT crate, card, region IDs
L1CaloRegionDetId::L1CaloRegionDetId(unsigned icrate, unsigned icard, unsigned irgn) :
  DetId(Calo, 2)
{

  int ieta=0;
  int iphi=0;

  // Calculate iphi
  int phi_index = icrate % 9;
  if ((icard == 0) || (icard == 2) || (icard == 4))
    phi_index = phi_index * 2;
  else if ((icard == 1) || (icard == 3) || (icard == 5))
    phi_index = phi_index * 2 + 1;
  else if (icard == 6)
    phi_index = phi_index * 2 + irgn;
  // for HF
  else if (icard == 999)
    phi_index = phi_index * 2 + (irgn/4);
  iphi = (22 - phi_index) % 18;

  // Calculate ieta
  int eta_index = 0;
  if (icard < 6)
    eta_index = (icard/2) * 2 + irgn;
  else if (icard == 6)
    eta_index = 6;
  // HF
  else if (icard == 999)
    eta_index = (irgn % 4) + 7;
  
  if (icrate < 9)
    ieta = 10 - eta_index;
  else if (icrate >= 9)
    ieta = 11 + eta_index;

  /// TODO - check calculation of ieta and iphi from RCT crate/card/region #
  id_ |= (ieta & 0x1f) | ((iphi & 0x1f)<<5);
}

// return RCT crate ID
unsigned L1CaloRegionDetId::rctCrate() const { // TODO - check this is correct!
  unsigned phiCrate = ((N_PHI + 4 - iphi()) % N_PHI) / 2;
  return (ieta()<(N_ETA/2) ? phiCrate : phiCrate + N_PHI/2) ;
}

// return RCT card number
unsigned L1CaloRegionDetId::rctCard() const {
  unsigned card = 999;
  unsigned rct_phi_index = (22 - iphi()) % 18;
  if ((ieta() == 4) || (ieta() == 17)){
    card = 6;
  }
  else if ((ieta() > 4) && (ieta() <= 10)){
    unsigned index = (ieta() - 5)/2;
    card = ((2 - index) * 2) + (rct_phi_index % 2);
  }
  else if ((ieta() >= 11) && (ieta() < 17)){
    unsigned index = (ieta() - 11)/2;
    card = (index * 2) + (rct_phi_index % 2);
  }
  return card;
}

// return RCT region number
unsigned L1CaloRegionDetId::rctRegion() const {
  unsigned rgn = 999;
  unsigned rct_phi_index = (22 - iphi()) % 18;
  if (ieta() < 4){
    rgn = (3 - ieta()) + 4 * (rct_phi_index % 2);
  }
  else if (ieta() > 17){
    rgn = (ieta() - 18) + 4 * (rct_phi_index % 2);
  }
  else if ((ieta() == 4) || (ieta() == 17)){
    rgn = (rct_phi_index % 2);
  }
  else if ((ieta() > 4) && (ieta() <= 10)){
    rgn = (ieta() % 2);
  }
  else if ((ieta() >= 11) && (ieta() < 17)){
    rgn = ((ieta() - 1) % 2);
  }
  return rgn;
}
