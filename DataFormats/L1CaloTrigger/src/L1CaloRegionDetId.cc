

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"

// default constructor - null DetId
L1CaloRegionDetId::L1CaloRegionDetId() : DetId() { }


// construct from raw id
L1CaloRegionDetId::L1CaloRegionDetId(uint32_t rawid) : DetId(rawid) { }


// construct from ieta, iphi indices
L1CaloRegionDetId::L1CaloRegionDetId(int ieta, int iphi) :
  DetId(Calo, 2) 
{ 
  id_ |= ((ieta>0)?(0x8|ieta):(-ieta)) |
    (iphi&0x1f)<<4;
}


// construct from RCT crate, card, region IDs
L1CaloRegionDetId::L1CaloRegionDetId(unsigned icrate, unsigned icard, unsigned irgn) :
  DetId(Calo, 2)
{

  // calculate ieta
  int ieta=0;

  // calculate iphi
  int iphi=0;

  id_ |= ((ieta>0)?(0x8|ieta):(-ieta)) |
    (iphi&0x1f)<<4;
}
