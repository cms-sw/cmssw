#ifndef L1TObjects_L1GctChannelMask_h
#define L1TObjects_L1GctChannelMask_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <ostream>

class L1GctChannelMask {
public:
  /// default constructor sets all masks to false
  L1GctChannelMask();
  ~L1GctChannelMask() {}

  /// mask EM candidates from an RCT crate
  void maskEmCrate(unsigned crate);

  /// mask a region
  void maskRegion(unsigned ieta, unsigned iphi);

  /// mask eta range from total Et sum
  void maskTotalEt(unsigned ieta);

  /// mask eta range from missing Et sum
  void maskMissingEt(unsigned ieta);

  /// mask eta range from total Ht sum
  void maskTotalHt(unsigned ieta);

  /// mask eta range from missing Ht sum
  void maskMissingHt(unsigned ieta);

  /// get EM masks for an RCT crate
  bool emCrateMask(unsigned crate) const;

  /// get region masks
  bool regionMask(unsigned ieta, unsigned iphi) const;

  // get total Et masks
  bool totalEtMask(unsigned ieta) const;

  // get missing Et masks
  bool missingEtMask(unsigned ieta) const;

  // get total Ht masks
  bool totalHtMask(unsigned ieta) const;

  // get missing Ht masks
  bool missingHtMask(unsigned ieta) const;

private:
  bool emCrateMask_[18];     // mask EM from RCT crate[n]
  bool regionMask_[22][18];  // mask region[ieta][iphi]
  bool tetMask_[22];
  bool metMask_[22];
  bool htMask_[22];
  bool mhtMask_[22];

  COND_SERIALIZABLE;
};

std::ostream& operator<<(std::ostream& os, const L1GctChannelMask obj);

#endif
