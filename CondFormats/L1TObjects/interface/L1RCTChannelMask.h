#ifndef L1TObjects_L1RCTChannelMask_h
#define L1TObjects_L1RCTChannelMask_h

struct L1RCTChannelMask {

  bool ecalMask[18][2][28];
  bool hcalMask[18][2][28];
  bool hfMask[18][2][4];

};

#endif
