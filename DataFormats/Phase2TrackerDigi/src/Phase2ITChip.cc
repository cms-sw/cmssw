#include <algorithm>
#include <map>
#include <tuple>
#include <utility>
#include <vector>
#include <string>
#include "DataFormats/Phase2TrackerDigi/interface/Phase2ITQCore.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2ITChip.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2ITDigiHit.h"
#include "DataFormats/Phase2TrackerDigi/interface/ChipModuleMap.h"
#include "EventFilter/Phase2PixelRawToDigi/interface/Phase2DAQFormatSpecification.h"

using namespace Phase2DAQFormatSpecification;

Phase2ITChip::Phase2ITChip(int rocnum, const std::vector<Phase2ITDigiHit> hl, uint32_t detId) {
  hitList_ = hl;
  rocnum_ = rocnum;
  detId_ = detId;
}

unsigned int Phase2ITChip::size() { return hitList_.size(); }

//Returns the position (row,col) of the 4x4 QCores that contains a hit
std::pair<int, int> Phase2ITChip::getQCorePos(Phase2ITDigiHit hit) {
  int row = hit.row() / 4;
  int col = hit.col() / 4;
  return {row, col};
}

int Phase2ITChip::encodeQCoreIndex(int row, int col) { return ((4 * (row % 4) + (col % 4) + 8) % 16); }

std::pair<int, int> Phase2ITChip::decodeQCoreIndex(int index) {
  int adjusted = (index + 8) % 16;
  return {adjusted / 4, adjusted % 4};
}

std::pair<int, int> Phase2ITChip::getGlobalPixelCoordinate(
    int chipId, int subtype, int qcoreCol, int qcoreRow, int localCol, int localRow, bool keepMode) {
  // DROP: Pixel hits in the gap are completely dropped from encoding.
  // AGGREGATE: Pixel hits in the gap are assigned to the boundary sensor IDs with ADCs being stacked up.
  // KEEP: Pixel hits in the gap are kept with its own sensor IDs.
  // FIXME Most realistic scenario will be AGGREGATE mode but need to check if the ADCs are plainly being stacked up
  const int rowsPerChip = keepMode ? kRowsPerChipKeep : kRowsPerChip;
  const int largePixelNRows = keepMode ? kLargePixelNRowsKeep : kLargePixelNRows;
  const int colsPerChip = keepMode ? kColsPerChipKeep : kColsPerChip;
  const int largePixelNCols = keepMode ? kLargePixelNColsKeep : kLargePixelNCols;

  // Recover the physical row/col half from (subtype, chipId) via ChipModuleMap.
  auto q = ChipModuleMap::quadrantOf(subtype, chipId);
  const int rowHalf = (q.first > 0) ? 1 : 0;
  const int colHalf = (q.second > 0) ? 1 : 0;
  int rowOffset = rowHalf ? (rowsPerChip + largePixelNRows) : 0;
  int colOffset = colHalf ? (colsPerChip + largePixelNCols) : 0;
  int globalRow = rowOffset + qcoreRow * HITMAP_COL + localRow;
  int globalCol = colOffset + qcoreCol * HITMAP_ROW + localCol;

  return {globalRow, globalCol};
}

//Takes in an oranized list of Phase2ITQCores and sets the islast and isneighbor properties of those qcores
std::vector<Phase2ITQCore> linkQCores(std::vector<Phase2ITQCore> qcores) {
  // First set islast flags
  if (!qcores.empty()) {
    for (size_t i = 0; i < qcores.size() - 1; i++) {
      if (qcores[i].getCol() != qcores[i + 1].getCol()) {
        qcores[i].setIsLast(true);
      }
    }
    qcores[qcores.size() - 1].setIsLast(true);
  }

  // Then set isneighbour flags. Per RD53B Sec 10.4:
  // isneighbour=1 means the previous qrow address is current_qrow - 1 (i.e. consecutive qrows within the same ccol), so the qrow field can be omitted on the wire.
  // Only candidates are pairs in the same ccol (previous not islast).
  for (size_t i = 1; i < qcores.size(); i++) {
    if (!qcores[i - 1].islast() && qcores[i].getRow() == qcores[i - 1].getRow() + 1) {
      qcores[i].setIsNeighbour(true);
    }
  }

  return qcores;
}

//Takes in a list of hits and organizes them into the 4x4 QCores that contain them.
//One pass: bucket hits by qcore position, build one Phase2ITQCore per bucket.
//std::map keeps buckets sorted by (row, col); we then sort by (col, row) which is the order linkQCores expects.
std::vector<Phase2ITQCore> Phase2ITChip::getOrganizedQCores() {
  // key = (qcoreRow, qcoreCol); value = (adcs[16], hits[16])
  std::map<std::pair<int, int>, std::pair<std::vector<int>, std::vector<int>>> byPos;
  for (const auto& hit : hitList_) {
    auto pos = getQCorePos(hit);
    auto it = byPos.find(pos);
    if (it == byPos.end()) {
      it = byPos.emplace(pos, std::make_pair(std::vector<int>(16, 0), std::vector<int>(16, 0))).first;
    }
    int i = encodeQCoreIndex(hit.row(), hit.col());
    it->second.first[i] = hit.adc();
    it->second.second[i] = 1;
  }

  std::vector<Phase2ITQCore> qcores;
  qcores.reserve(byPos.size());
  for (auto& [pos, payload] : byPos) {
    qcores.emplace_back(0, pos.second, pos.first, false, false, payload.first, payload.second);
  }

  std::sort(qcores.begin(), qcores.end(), [](const Phase2ITQCore& a, const Phase2ITQCore& b) {
    return std::make_pair(a.getCol(), a.getRow()) < std::make_pair(b.getCol(), b.getRow());
  });

  return linkQCores(std::move(qcores));
}

//Returns the encoding of the readout chip
Phase2ITBitBuffer Phase2ITChip::getChipCode(bool dropTot) {
  Phase2ITBitBuffer code;

  if (!hitList_.empty()) {
    std::vector<Phase2ITQCore> qcores = getOrganizedQCores();
    bool isNewCol = true;

    for (auto& qcore : qcores) {
      qcore.encodeQCore(code, isNewCol, dropTot);
      isNewCol = qcore.islast();
    }
  }

  return code;
}
