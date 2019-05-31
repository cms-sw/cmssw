#ifndef L1TObjects_L1RCTChannelMask_h
#define L1TObjects_L1RCTChannelMask_h
#include "CondFormats/Serialization/interface/Serializable.h"

#include <ostream>

struct L1RCTChannelMask {
  bool ecalMask[18][2][28];
  bool hcalMask[18][2][28];
  bool hfMask[18][2][4];
  void print(std::ostream& s) const {
    s << "Printing record L1RCTChannelMaskRcd " << std::endl;
    s << "Masked channels in L1RCTChannelMask" << std::endl;
    for (int i = 0; i < 18; i++)
      for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 28; k++) {
          if (ecalMask[i][j][k])
            s << "ECAL masked channel: RCT crate " << i << " iphi " << j << " ieta " << k << std::endl;
          if (hcalMask[i][j][k])
            s << "HCAL masked channel: RCT crate " << i << " iphi " << j << " ieta " << k << std::endl;
        }
        for (int k = 0; k < 4; k++)
          if (hfMask[i][j][k])
            s << "HF masked channel: RCT crate " << i << " iphi " << j << " ieta " << k << std::endl;
      }
  }

  COND_SERIALIZABLE;
};

#endif
