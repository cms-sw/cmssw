#ifndef DataFormats_Phase2TrackerDigi_ChipModuleMap_h
#define DataFormats_Phase2TrackerDigi_ChipModuleMap_h

// A module's pixels are split into chips on a grid of (row-half, col-half).
// With the IT axis convention x = row, y = column, each chip occupies a quadrant (x_sign, y_sign) with signs in {-1, 0, +1}
// (0 = no split along that axis, i.e. 1- or 2-chip modules)
//
//   packing step uses chipIndex() : (subtype, x_sign, y_sign) -> chip index
//   unpacking step uses quadrantOf() : (subtype, chip index) -> (x_sign, y_sign)
// which are exact inverses.

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ChipModuleMap {

  // Chip index -> (x_sign, y_sign) quadrant, keyed by Module_SubType (1..12).
  inline const std::map<int, std::vector<std::pair<int, int>>> CHIP_QUADRANT = {
      {1, {{0, 0}}},                                   // TBPX L1   : 1 chip
      {2, {{0, +1}, {0, -1}}},                         // TBPX L2   : 2 chips
      {3, {{+1, +1}, {+1, -1}, {-1, -1}, {-1, +1}}},   // TBPX L3
      {4, {{+1, +1}, {+1, -1}, {-1, -1}, {-1, +1}}},   // TBPX L4
      {5, {{0, -1}, {0, +1}}},                         // TFPX R1/R2: 2 chips
      {6, {{+1, -1}, {+1, +1}, {-1, -1}, {-1, +1}}},   // TFPX R3   (Y-flipped)
      {7, {{+1, -1}, {+1, +1}, {-1, -1}, {-1, +1}}},   // TFPX R4   (Y-flipped)
      {8, {{+1, +1}, {+1, -1}, {-1, -1}, {-1, +1}}},   // TEPX R1
      {9, {{+1, +1}, {+1, -1}, {-1, -1}, {-1, +1}}},   // TEPX R2
      {10, {{+1, +1}, {+1, -1}, {-1, -1}, {-1, +1}}},  // TEPX R2/R3
      {11, {{+1, +1}, {+1, -1}, {-1, -1}, {-1, +1}}},  // TEPX R4
      {12, {{+1, +1}, {+1, -1}, {-1, -1}, {-1, +1}}},  // TEPX R5
  };

  // Number of chips on a module of the given Module_SubType.
  // Throws std::runtime_error if the subtype is not in the table.
  inline int nChips(int subtype) {
    auto row = CHIP_QUADRANT.find(subtype);
    if (row == CHIP_QUADRANT.end())
      throw std::runtime_error("ChipModuleMap: unknown Module_SubType " + std::to_string(subtype));
    return static_cast<int>(row->second.size());
  }

  // Map a grid cell (row-half, col-half) to its signed (x, y) quadrant.
  // A module with a single chip along an axis yields sign 0 on that axis.
  inline std::pair<int, int> gridToQuadrant(int ichipRow, int ichipCol, int nChipRows, int nChipCols) {
    int x = (nChipRows == 1) ? 0 : (ichipRow ? +1 : -1);
    int y = (nChipCols == 1) ? 0 : (ichipCol ? +1 : -1);
    return {x, y};
  }

  // Quadrant -> chip index for a given Module_SubType (packing direction).
  // Throws std::runtime_error on unknown subtype or quadrant.
  inline int chipIndex(int subtype, int xSign, int ySign) {
    auto row = CHIP_QUADRANT.find(subtype);
    if (row == CHIP_QUADRANT.end())
      throw std::runtime_error("ChipModuleMap: unknown Module_SubType " + std::to_string(subtype));
    auto it = std::find(row->second.begin(), row->second.end(), std::make_pair(xSign, ySign));
    if (it == row->second.end())
      throw std::runtime_error("ChipModuleMap: quadrant (" + std::to_string(xSign) + "," + std::to_string(ySign) +
                               ") not in subtype " + std::to_string(subtype));
    return static_cast<int>(std::distance(row->second.begin(), it));
  }

  // Chip index -> quadrant for a given Module_SubType (unpacking direction).
  // Exact inverse of chipIndex(). Throws std::runtime_error on unknown subtype or out-of-range chip index.
  inline std::pair<int, int> quadrantOf(int subtype, int chip) {
    auto row = CHIP_QUADRANT.find(subtype);
    if (row == CHIP_QUADRANT.end())
      throw std::runtime_error("ChipModuleMap: unknown Module_SubType " + std::to_string(subtype));
    if (chip < 0 || chip >= static_cast<int>(row->second.size()))
      throw std::runtime_error("ChipModuleMap: chip index " + std::to_string(chip) + " out of range for subtype " +
                               std::to_string(subtype));
    return row->second[static_cast<size_t>(chip)];
  }

}  // namespace ChipModuleMap

#endif  // DataFormats_Phase2TrackerDigi_ChipModuleMap_h
