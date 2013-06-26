#include "CondFormats/L1TObjects/interface/L1GctChannelMask.h"

L1GctChannelMask::L1GctChannelMask() {
  for (unsigned i=0; i<18; ++i) {
    emCrateMask_[i] = false;
  }
  for (unsigned ieta=0; ieta<22; ++ieta) {
    for (unsigned iphi=0; iphi<18; ++iphi) {
      regionMask_[ieta][iphi] = false;
    }
  }
  for (unsigned i=0; i<22; ++i) {
    tetMask_[i] = false;
    metMask_[i] = false;
    htMask_[i] = false;
    mhtMask_[i] = false;
  }
}


void L1GctChannelMask::maskEmCrate(unsigned crate) {
  if (crate < 18) emCrateMask_[crate] = true;
}


void L1GctChannelMask::maskRegion(unsigned ieta, unsigned iphi) {
  if (ieta < 22 && iphi < 18) regionMask_[ieta][iphi] = true;
}


void L1GctChannelMask::maskTotalEt(unsigned ieta) {
  if (ieta < 22) tetMask_[ieta] = true;
}


void L1GctChannelMask::maskMissingEt(unsigned ieta) {
  if (ieta < 22) metMask_[ieta] = true;
}


void L1GctChannelMask::maskTotalHt(unsigned ieta) {
  if (ieta < 22) htMask_[ieta] = true;
}


void L1GctChannelMask::maskMissingHt(unsigned ieta) {
  if (ieta < 22) mhtMask_[ieta] = true;
}


bool L1GctChannelMask::emCrateMask(unsigned crate) const {
  if (crate < 18) { return emCrateMask_[crate]; }
  else return true;
}


bool L1GctChannelMask::regionMask(unsigned ieta, unsigned iphi) const {
  if (ieta < 22 && iphi < 18) { return regionMask_[ieta][iphi]; }
  else return true;
}


bool L1GctChannelMask::totalEtMask(unsigned ieta) const {
  if (ieta < 22) return tetMask_[ieta];
  else return true;
}


bool L1GctChannelMask::missingEtMask(unsigned ieta) const {
  if (ieta < 22) return metMask_[ieta];
  else return true;
}


bool L1GctChannelMask::totalHtMask(unsigned ieta) const {
  if (ieta < 22) return htMask_[ieta];
  else return true;
}


bool L1GctChannelMask::missingHtMask(unsigned ieta) const {
  if (ieta < 22) return mhtMask_[ieta];
  else return true;
}

std::ostream& operator << (std::ostream& os, const L1GctChannelMask obj) {
  os << "L1GctChannelMask :" << std::endl;

  // get masks without changing interface, sigh
  unsigned emCrateMask(0), tetMask(0), metMask(0), htMask(0), mhtMask(0);
  for (unsigned i=0; i<18; ++i) {
    emCrateMask |= ((obj.emCrateMask(i)?1:0)<<i);
  }

  for (unsigned i=0; i<22; ++i) {
    tetMask |= ((obj.totalEtMask(i)?1:0)<<i);
    metMask |= ((obj.missingEtMask(i)?1:0)<<i);
    htMask |= ((obj.totalHtMask(i)?1:0)<<i);
    mhtMask |= ((obj.missingHtMask(i)?1:0)<<i);
  }

   os << "  EM crate mask    = " << std::hex << emCrateMask << std::endl;
   os << "  EtTot mask       = " << std::hex << tetMask << std::endl;
   os << "  EtMiss mask      = " << std::hex << metMask << std::endl;
   os << "  HtTot mask       = " << std::hex << htMask << std::endl;
   os << "  HtMiss mask      = " << std::hex << mhtMask << std::endl;
  
   for (unsigned ieta=0; ieta<22; ++ieta) {
    for (unsigned iphi=0; iphi<18; ++iphi) {
      if ( obj.regionMask(ieta, iphi) )  {
	os << "  Region mask      : " << std::dec << ieta << ", " << iphi << std::endl;
      }
    }
  }
  return os;
}
