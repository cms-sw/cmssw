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
