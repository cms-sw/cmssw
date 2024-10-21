#ifndef DataFormats_Phase2TrackerDigi_ReadoutChip_H
#define DataFormats_Phase2TrackerDigi_ReadoutChip_H
#include <vector>
#include <utility>
#include <string>
#include "DataFormats/Phase2TrackerDigi/interface/QCore.h"
#include "DataFormats/Phase2TrackerDigi/interface/DigiHitRecord.h"

class ReadoutChip {
  std::vector<DigiHitRecord> hitList;
  int rocnum_;

public:
  ReadoutChip(int rocnum, std::vector<DigiHitRecord> hl);

  unsigned int size();
  int rocnum() const { return rocnum_; }

  std::vector<QCore> get_organized_QCores();
  std::vector<bool> get_chip_code();

private:
  std::pair<int, int> get_QCore_pos(DigiHitRecord hit);

  QCore get_QCore_from_hit(DigiHitRecord pixel);
  std::vector<QCore> rem_duplicates(std::vector<QCore> qcores);
  std::vector<QCore> organize_QCores(std::vector<QCore> qcores);
};

#endif  // DataFormats_Phase2TrackerDigi_ReadoutChip_H
