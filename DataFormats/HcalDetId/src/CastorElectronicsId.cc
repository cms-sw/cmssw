#include "DataFormats/HcalDetId/interface/CastorElectronicsId.h"


CastorElectronicsId::CastorElectronicsId() {
  castorElectronicsId_=0xffffffffu;
}

CastorElectronicsId::CastorElectronicsId(uint32_t id) {
  castorElectronicsId_=id;
}

CastorElectronicsId::CastorElectronicsId(int fiberChan, int fiberIndex, int spigot, int dccid) {
  castorElectronicsId_=(fiberChan&0x3) | (((fiberIndex-1)&0xf)<<2) |
    ((spigot&0xF)<<6) | ((dccid&0xF)<<10);
}

CastorElectronicsId::CastorElectronicsId(int slbChan, int slbSite, int spigot, int dccid, int crate, int slot, int tb) {
  castorElectronicsId_=(slbChan&0x3) | (((slbSite-1)&0xf)<<2) |
    ((spigot&0xF)<<6) | ((dccid&0xF)<<10);
  castorElectronicsId_|=((tb&0x1)<<19) | ((slot&0x1f)<<14) | ((crate&0x3f)<<20);
  castorElectronicsId_|=0x02000000;
}

std::string CastorElectronicsId::slbChannelCode() const {
  std::string retval;
  if (isTriggerChainId()) {
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

void CastorElectronicsId::setHTR(int crate, int slot, int tb) {
  castorElectronicsId_&=0x3FFF; // keep the readout chain info
  castorElectronicsId_|=((tb&0x1)<<19) | ((slot&0x1f)<<14) | ((crate&0x3f)<<20);
}

std::ostream& operator<<(std::ostream& os,const CastorElectronicsId& id) {
  if (id.isTriggerChainId()) {
    return os << id.dccid() << ',' << id.spigot() << ",SLB" << id.slbSiteNumber() << ',' << id.slbChannelIndex() << " (HTR "
	      << id.readoutVMECrateId() << ":" << id.htrSlot() << ((id.htrTopBottom()==1)?('t'):('b')) << ')'; 
    
  } else {
    return os << id.dccid() << ',' << id.spigot() << ',' << id.fiberIndex() << ',' << id.fiberChanId() << " (HTR "
	      << id.readoutVMECrateId() << ":" << id.htrSlot() << ((id.htrTopBottom()==1)?('t'):('b')) << ')'; 
  }
}


