#ifndef EventFilter_Phase2PixelRawToDigi_ELinkChipMap_h
#define EventFilter_Phase2PixelRawToDigi_ELinkChipMap_h

// Chip -> elink fan-out, keyed by Module_SubType (the subtype uniquely
// determines N_ELinks). Each entry is `chip_index -> list of elink indices
// the chip drives`; the chip index follows the ChipModuleMap quadrant
// convention (DataFormats/Phase2TrackerDigi/interface/ChipModuleMap.h).

#include <vector>

class ELinkChipMap {
public:
  using ChipToElinks = std::vector<std::vector<int>>;

  static ChipToElinks resolveFanout(int subtype);

  // 1 + max elink index in the fan-out (i.e. number of distinct elinks used).
  static int numElinks(const ChipToElinks& f);
};

#endif  // EventFilter_Phase2PixelRawToDigi_ELinkChipMap_h
