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

HcalElectronicsId::HcalElectronicsId(int slbChan, int slbSite, int spigot, int dccid, int crate, int slot, int tb) {
  hcalElectronicsId_=(slbChan&0x3) | (((slbSite)&0x7)<<2) |
    ((spigot&0xF)<<5) | ((dccid&0x1F)<<9);
  hcalElectronicsId_|=((tb&0x1)<<19) | ((slot&0x1f)<<14) | ((crate&0x3f)<<20);
  hcalElectronicsId_|=0x02000000;
}

HcalElectronicsId::HcalElectronicsId(int crate, int slot, int fiber, int fc, bool isTrigger) {
  hcalElectronicsId_=(fc&0xF) | (((fiber)&0x1F)<<4) |
    ((slot&0xF)<<9) | ((crate&0x3F)<<13);
  if (isTrigger)   hcalElectronicsId_|=0x02000000;
  hcalElectronicsId_|=0x04000000;
}

std::string HcalElectronicsId::slbChannelCode() const {
  std::string retval;
  if (isTriggerChainId() && isVMEid()) {
    if (htrTopBottom()) { // top
      switch (slbChannelIndex()) {
      case (0): retval="A0"; break;
      case (1): retval="A1"; break;
      case (2): retval="C0"; break;
      case (3): retval="C1"; break;
      }
    } else {
      switch (slbChannelIndex()) {
      case (0): retval="B0"; break;
      case (1): retval="B1"; break;
      case (2): retval="D0"; break;
      case (3): retval="D1"; break;
      }
    }
  }
  return retval;
}

void HcalElectronicsId::setHTR(int crate, int slot, int tb) {
  if (isUTCAid()) return; // cannot do this for uTCA
  hcalElectronicsId_&=0x3FFF; // keep the readout chain info
  hcalElectronicsId_|=((tb&0x1)<<19) | ((slot&0x1f)<<14) | ((crate&0x3f)<<20);
}

std::ostream& operator<<(std::ostream& os,const HcalElectronicsId& id) {
  if (id.isUTCAid()) {
    if (id.isTriggerChainId()) os << "UTCA(trigger): ";
    else os << "UTCA: ";
    return os << id.crateId() << ',' << id.slot() << ',' << id.fiberIndex() << ',' << id.fiberChanId();
  } else {
    if (id.isTriggerChainId()) {
      return os << id.dccid() << ',' << id.spigot() << ",SLB" << id.slbSiteNumber() << ',' << id.slbChannelIndex() << " (HTR "
		<< id.readoutVMECrateId() << ":" << id.htrSlot() << ((id.htrTopBottom()==1)?('t'):('b')) << ')'; 
      
    } else {
      return os << id.dccid() << ',' << id.spigot() << ',' << id.fiberIndex() << ',' << id.fiberChanId() << " (HTR "
		<< id.readoutVMECrateId() << ":" << id.htrSlot() << ((id.htrTopBottom()==1)?('t'):('b')) << ')'; 
    }
  }
}


