#include "DataFormats/HcalDetId/interface/HcalElectronicsId.h"

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
