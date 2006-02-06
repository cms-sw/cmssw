#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"


HcalElectronicsId::HcalElectronicsId() {
  hcalElectronicsId_=0xffffffffu;
}

HcalElectronicsId::HcalElectronicsId(uint32_t id) {
  hcalElectronicsId_=id;
}

HcalElectronicsId::HcalElectronicsId(int fiberChan, int fiberIndex, int spigot, int dccid) {
  hcalElectronicsId_=(fiberChan&0x3) | (((fiberIndex-1)&0x7)<<2) |
    ((spigot&0xF)<<5) | ((dccid&0x1F)<<9);
}

void HcalElectronicsId::setHTR(int crate, int slot, int tb) {
  hcalElectronicsId_&=0x3FFF; // keep the readout chain info
  hcalElectronicsId_|=((tb&0x1)<<19) | ((slot&0x1f)<<14) | ((crate&0x3f)<<20);
}

std::ostream& operator<<(std::ostream& os,const HcalElectronicsId& id) {
  return os << id.dccid() << ',' << id.spigot() << ',' << id.fiberIndex() << ',' << id.fiberChanId() << " (HTR "
	    << id.readoutVMECrateId() << ":" << id.htrSlot() << ((id.htrTopBottom()==1)?('t'):('b')) << ')'; 
}


