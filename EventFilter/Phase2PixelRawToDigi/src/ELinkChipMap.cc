#include "EventFilter/Phase2PixelRawToDigi/interface/ELinkChipMap.h"

#include <map>

namespace {

  // Chip -> elink fan-out, keyed by Module_SubType alone (the subtype uniquely determines N_ELinks).
  // The chip index follows the per-subtype chip -> quadrant order defined in ChipModuleMap::CHIP_QUADRANT, which differs between detector sections
  // (e.g. TFPX R3/R4 are Y-flipped with respect to TBPX).
  // Each row below is written in that subtype's own chip order.
  const std::map<int, ELinkChipMap::ChipToElinks> kFanoutByType = {
      {1, {{0, 1, 2}}},            // TBPX L1   : 1 chip broadcast on 3 elinks
      {2, {{0}, {1}}},             // TBPX L2   : 2 chips, 2 elinks
      {3, {{0}, {0}, {1}, {1}}},   // TBPX L3   : row-pair
      {4, {{0}, {0}, {0}, {0}}},   // TBPX L4   : 4 chips share 1 elink
      {5, {{0}, {1, 2}}},          // TFPX R1/R2: chip 0->e0, chip 1->e1&e2
      {6, {{0}, {0}, {1}, {1}}},   // TFPX R3
      {7, {{0}, {0}, {1}, {1}}},   // TFPX R4
      {8, {{0}, {1}, {2}, {3}}},   // TEPX R1   : 4 chips dedicated, 4 elinks
      {9, {{2}, {0}, {0}, {1}}},   // TEPX R2   : 3 elinks (ChipModuleMap order; e0 = the two y- chips)
      {10, {{0}, {0}, {1}, {1}}},  // TEPX R2/R3: 2 elinks (row-pair)
      {11, {{0}, {1}, {0}, {1}}},  // TEPX R4   : 2 elinks (ChipModuleMap order; diagonal pairs share an elink)
      {12, {{0}, {0}, {0}, {0}}},  // TEPX R5   : 4 chips share 1 elink
  };

}  // namespace

ELinkChipMap::ChipToElinks ELinkChipMap::resolveFanout(int subtype) { return kFanoutByType.at(subtype); }

int ELinkChipMap::numElinks(const ChipToElinks& f) {
  int maxE = -1;
  for (const auto& v : f)
    for (int e : v)
      if (e > maxE)
        maxE = e;
  return maxE + 1;
}
