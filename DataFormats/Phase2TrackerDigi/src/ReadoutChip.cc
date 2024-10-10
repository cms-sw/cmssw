#include <vector>
#include <utility>
#include <string>
#include <iostream>
#include "DataFormats/Phase2TrackerDigi/interface/QCore.h"
#include "DataFormats/Phase2TrackerDigi/interface/ReadoutChip.h"
#include "DataFormats/Phase2TrackerDigi/interface/DigiHitRecord.h"

ReadoutChip::ReadoutChip(int rocnum, std::vector<DigiHitRecord> hl) {
  hitList = hl;
  rocnum_ = rocnum;
}

unsigned int ReadoutChip::size() { return hitList.size(); }

//Returns the position (row,col) of the 4x4 QCores that contains a hit
std::pair<int, int> ReadoutChip::get_QCore_pos(DigiHitRecord hit) {
  int row = hit.row() / 4;
  int col = hit.col() / 4;
  return {row, col};
}

//Takes a hit and returns the 4x4 QCore that contains it
QCore ReadoutChip::get_QCore_from_hit(DigiHitRecord pixel) {
  std::vector<int> adcs(16, 0), hits(16, 0);
  std::pair<int, int> pos = get_QCore_pos(pixel);

  for (const auto& hit : hitList) {
    if (get_QCore_pos(hit) == pos) {
      int i = (4 * (hit.row() % 4) + (hit.col() % 4) + 8) % 16;
      adcs[i] = hit.adc();
      hits[i] = 1;
    }
  }

  QCore qcore(0, pos.second, pos.first, false, false, adcs, hits);
  return qcore;
}

//Removes duplicated QCores
std::vector<QCore> ReadoutChip::rem_duplicates(std::vector<QCore> qcores) {
  std::vector<QCore> list = {};

  size_t i = 0;
  while (i < qcores.size()) {
    for (size_t j = i + 1; j < qcores.size();) {
      if (qcores[j].get_col() == qcores[i].get_col() && qcores[j].get_row() == qcores[i].get_row()) {
        qcores.erase(qcores.begin() + j);
      } else {
        ++j;
      }
    }
    list.push_back(qcores[i]);
    ++i;
  }

  return list;
}

//Returns a list of the qcores with hits arranged by increasing column and then row numbers
std::vector<QCore> ReadoutChip::organize_QCores(std::vector<QCore> qcores) {
  std::vector<QCore> organized_list = {};
  while (qcores.size() > 0) {
    int min = 0;

    for (size_t i = 1; i < qcores.size(); i++) {
      if (qcores[i].get_col() < qcores[min].get_col()) {
        min = i;
      } else if (qcores[i].get_col() == qcores[min].get_col() && qcores[i].get_row() < qcores[min].get_row()) {
        min = i;
      }
    }

    organized_list.push_back(qcores[min]);
    qcores.erase(qcores.begin() + min);
  }

  return organized_list;
}

//Takes in an oranized list of QCores and sets the islast and isneighbor properties of those qcores
std::vector<QCore> link_QCores(std::vector<QCore> qcores) {
  for (size_t i = 1; i < qcores.size(); i++) {
    if (qcores[i].get_row() == qcores[i - 1].get_row()) {
      qcores[i].setIsNeighbour(true);
    }
  }

  //.size() is unsigned. If size is zero size()-1 is a huge number hence this needs to be protected
  if (qcores.size() > 0) {
    for (size_t i = 0; i < qcores.size() - 1; i++) {
      if (qcores[i].get_col() != qcores[i + 1].get_col()) {
        qcores[i].setIsLast(true);
      }
    }
    qcores[qcores.size() - 1].setIsLast(true);
  }

  return qcores;
}

//Takes in a list of hits and organizes them into the 4x4 QCores that contains them
std::vector<QCore> ReadoutChip::get_organized_QCores() {
  std::vector<QCore> qcores = {};

  for (const auto& hit : hitList) {
    qcores.push_back(get_QCore_from_hit(hit));
  }

  return (link_QCores(organize_QCores(rem_duplicates(qcores))));
}

//Returns the encoding of the readout chip
std::vector<bool> ReadoutChip::get_chip_code() {
  std::vector<bool> code = {};

  if (hitList.size() > 0) {
    std::vector<QCore> qcores = get_organized_QCores();
    bool is_new_col = true;

    for (auto& qcore : qcores) {
      std::vector<bool> qcore_code = qcore.encodeQCore(is_new_col);
      code.insert(code.end(), qcore_code.begin(), qcore_code.end());

      is_new_col = qcore.islast();
    }
  }

  return code;
}
