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

bool L1GctChannelMask::emCrateMask(unsigned crate) {
  if (crate < 18) { return emCrateMask_[crate]; }
  else return true;
}

bool L1GctChannelMask::regionMask(unsigned ieta, unsigned iphi) {
  if (ieta < 22 && iphi < 18) { return regionMask_[ieta][iphi]; }
  else return true;
}
