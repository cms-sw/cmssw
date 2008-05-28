#ifndef L1TObjects_L1GctChannelMask_h
#define L1TObjects_L1GctChannelMask_h

class L1GctChannelMask {
 public:
  
  /// default constructor sets all masks to false
  L1GctChannelMask();
  ~L1GctChannelMask();

  /// mask EM candidates from an RCT crate
  void maskEmCrate(unsigned crate);

  /// mask a region
  void maskRegion(unsigned ieta, unsigned iphi);

  /// get EM masks for an RCT crate
  bool emCrateMask(unsigned crate);

  /// get region masks
  bool regionMask(unsigned ieta, unsigned iphi);

 private:
  bool emCrateMask_[18];     // mask EM from RCT crate[n]
  bool regionMask_[22][18];  // mask region[ieta][iphi]

};

#endif


