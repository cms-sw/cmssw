

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
L1CaloRegionDetId::L1CaloRegionDetId(bool isForward, unsigned icrate, unsigned icard, unsigned irgn) :
  DetId(Calo, 2)
{

  int ieta=0;
  int iphi=0;

  /// TODO - calculate ieta and iphi
  id_ |= (ieta & 0x1f) | ((iphi & 0x1f)<<5);
}

// return RCT crate ID
unsigned L1CaloRegionDetId::rctCrate() const { // TODO - check this is correct!
  unsigned phiCrate = ((N_PHI + 4 - iphi()) % N_PHI) / 2;
  return (ieta()<(N_ETA/2) ? phiCrate : phiCrate + N_PHI/2) ;
}
