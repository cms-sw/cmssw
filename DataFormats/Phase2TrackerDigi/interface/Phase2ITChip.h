#ifndef DataFormats_Phase2TrackerDigi_Phase2ITChip_H
#define DataFormats_Phase2TrackerDigi_Phase2ITChip_H
#include <vector>
#include <utility>
#include <string>
#include "DataFormats/Phase2TrackerDigi/interface/Phase2ITBitBuffer.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2ITQCore.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2ITDigiHit.h"

class Phase2ITChip {
  // Quarter cores collected into a chip (only active quarter cores with hits gets collected)

public:
  Phase2ITChip(int rocnum, const std::vector<Phase2ITDigiHit> hl, uint32_t detId);

  unsigned int size();
  int rocnum() const { return rocnum_; }
  uint32_t detId() const { return detId_; }

  std::vector<Phase2ITQCore> getOrganizedQCores();
  // dropTot=true forwards to Phase2ITQCore::encodeQCore to suppress ToTs
  Phase2ITBitBuffer getChipCode(bool dropTot = false);

  static int encodeQCoreIndex(int row, int col);
  static std::pair<int, int> decodeQCoreIndex(int index);

  // subtype = Module_SubType; selects the ChipModuleMap chip-index convention used to recover the (row, col) offset from chipId.
  static std::pair<int, int> getGlobalPixelCoordinate(
      int chipId, int subtype, int qcoreCol, int qcoreRow, int localCol, int localRow, bool keepMode = false);

  static constexpr int kColsPerROC = 54;
  static constexpr int kRowsPerChip = 672;
  static constexpr int kColsPerChip = 216;
  static constexpr int kLargePixelNRows = 10;
  static constexpr int kLargePixelNCols = 2;
  // KEEP-mode: Each chip boundary moves to the midline of the gap so that the gap is effectively none, making the pixels to be placed in one of the chips everytime.
  static constexpr int kRowsPerChipKeep = kRowsPerChip + kLargePixelNRows / 2;  // 677
  static constexpr int kColsPerChipKeep = kColsPerChip + kLargePixelNCols / 2;  // 217
  static constexpr int kLargePixelNRowsKeep = 0;
  static constexpr int kLargePixelNColsKeep = 0;

private:
  std::vector<Phase2ITDigiHit> hitList_;
  int rocnum_;
  uint32_t detId_;

  std::pair<int, int> getQCorePos(Phase2ITDigiHit hit);
};

#endif  // DataFormats_Phase2TrackerDigi_Phase2ITChip_H
