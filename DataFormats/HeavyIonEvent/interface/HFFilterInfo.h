#ifndef DataFormats_HeavyIonEvent_HFFilterInfo_H
#define DataFormats_HeavyIonEvent_HFFilterInfo_H

namespace reco {
  struct HFFilterInfo {
    unsigned short int numMinHFTowers2;
    unsigned short int numMinHFTowers3;
    unsigned short int numMinHFTowers4;
    unsigned short int numMinHFTowers5;
  };
}  // namespace reco

#endif
