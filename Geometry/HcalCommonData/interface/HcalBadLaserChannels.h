#ifndef Geometry_HcalCommonData_HcalBadLaserChannels_h
#define Geometry_HcalCommonData_HcalBadLaserChannels_h

#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

class HcalBadLaserChannels {

public:
  HcalBadLaserChannels() {}

  static int badChannelsHBHE () { return 72*3; }
  static int badChannelsHF()    { return 0; }
  static bool badChannelHBHE(HcalDetId id) {
    bool isbad(false);
    // Three RBX's in HB do not receive any laser light (HBM5, HBM8, HBM9)
    // They correspond to iphi = 15:18, 27:30, 31:34 respectively and
    // ieta < 0
    if (id.subdet()==HcalBarrel && id.ieta()<0) {
      if      (id.iphi()>=15 && id.iphi()<=18) isbad = true;
      else if (id.iphi()>=27 && id.iphi()<=34) isbad = true;
    }
    return isbad;
  }
};

#endif
