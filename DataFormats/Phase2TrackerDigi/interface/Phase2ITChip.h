#ifndef DataFormats_Phase2TrackerDigi_Phase2ITChip_H
#define DataFormats_Phase2TrackerDigi_Phase2ITChip_H
#include <vector>
#include <utility>
#include <string>
#include "DataFormats/Phase2TrackerDigi/interface/Phase2ITQCore.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2ITDigiHit.h"

class Phase2ITChip {
  // Quarter cores collected into a chip (only active quarter cores with hits gets collected)

public:
  Phase2ITChip(int rocnum, const std::vector<Phase2ITDigiHit> hl);

  unsigned int size();
  int rocnum() const { return rocnum_; }

  std::vector<Phase2ITQCore> get_organized_QCores();
  std::vector<bool> get_chip_code();

private:
  std::vector<Phase2ITDigiHit> hitList_;
  int rocnum_;

  std::pair<int, int> get_QCore_pos(Phase2ITDigiHit hit);

  Phase2ITQCore get_QCore_from_hit(Phase2ITDigiHit pixel);
  std::vector<Phase2ITQCore> rem_duplicates(std::vector<Phase2ITQCore> qcores);
  std::vector<Phase2ITQCore> organize_QCores(std::vector<Phase2ITQCore> qcores);
};

#endif  // DataFormats_Phase2TrackerDigi_Phase2ITChip_H
