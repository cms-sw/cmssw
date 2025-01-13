#include "DataFormats/Phase2TrackerDigi/interface/Phase2ITQCore.h"
#include <cmath>
#include <vector>
#include <iostream>

//4x4 region of hits in sensor coordinates
Phase2ITQCore::Phase2ITQCore(int rocid,
                             int ccol_in,
                             int qcrow_in,
                             bool isneighbour_in,
                             bool islast_in,
                             const std::vector<int>& adcs_in,
                             const std::vector<int>& hits_in) {
  rocid_ = rocid;
  ccol_ = ccol_in;
  qcrow_ = qcrow_in;
  isneighbour_ = isneighbour_in;
  islast_ = islast_in;
  adcs_ = adcs_in;
  hits_ = hits_in;
}

//Takes a hitmap in sensor coordinates in 4x4 and converts it to readout chip coordinates with 2x8
std::vector<bool> Phase2ITQCore::toRocCoordinates(std::vector<bool>& hitmap) {
  std::vector<bool> ROC_hitmap(16, false);

  for (size_t i = 0; i < hitmap.size(); i++) {
    int row = i / 4;
    int col = i % 4;
    int new_row;
    int new_col;

    if (row % 2 == 0) {
      new_row = row / 2;
      new_col = 2 * col;
    } else {
      new_row = row / 2;
      new_col = 2 * col + 1;
    }

    int new_index = 8 * new_row + new_col;
    ROC_hitmap[new_index] = hitmap[i];
  }

  return ROC_hitmap;
}

//Returns the hitmap for the Phase2ITQCore in 4x4 sensor coordinates
std::vector<bool> Phase2ITQCore::getHitmap() {
  std::vector<bool> hitmap = {};

  hitmap.reserve(hits_.size());
  for (auto hit : hits_) {
    hitmap.push_back(hit > 0);
  }

  return (toRocCoordinates(hitmap));
}

std::vector<int> Phase2ITQCore::getADCs() {
  std::vector<int> adcmap = {};

  adcmap.reserve(adcs_.size());
  for (auto adc : adcs_) {
    adcmap.push_back(adc);
  }

  return adcmap;
}

//Converts an integer into a binary, and formats it with the given length
std::vector<bool> Phase2ITQCore::intToBinary(int num, int length) {
  std::vector<bool> bi_num(length, false);

  for (int i = 0; i < length; ++i) {
    // Extract the (length - 1 - i)th bit from num
    bi_num[i] = (num >> (length - 1 - i)) & 1;
  }

  return bi_num;
}

//Takes a hitmap and returns true if it contains any hits
bool Phase2ITQCore::containsHit(std::vector<bool>& hitmap) {
  bool foundHit = false;
  for (size_t i = 0; i < hitmap.size(); i++) {
    if (hitmap[i]) {
      foundHit = true;
      break;
    }
  }

  return foundHit;
}

//Returns the Huffman encoded hitmap, created iteratively within this function
std::vector<bool> Phase2ITQCore::getHitmapCode(std::vector<bool> hitmap) {
  std::vector<bool> code = {};
  // If hitmap is a single bit, there is no need to further split the bits
  if (hitmap.size() == 1) {
    return code;
  }

  std::vector<bool> left_hitmap = std::vector<bool>(hitmap.begin(), hitmap.begin() + hitmap.size() / 2);
  std::vector<bool> right_hitmap = std::vector<bool>(hitmap.begin() + hitmap.size() / 2, hitmap.end());

  bool hit_left = containsHit(left_hitmap);
  bool hit_right = containsHit(right_hitmap);

  if (hit_left && hit_right) {
    code.push_back(true);
    code.push_back(true);

    std::vector<bool> left_code = getHitmapCode(left_hitmap);
    std::vector<bool> right_code = getHitmapCode(right_hitmap);

    code.insert(code.end(), left_code.begin(), left_code.end());
    code.insert(code.end(), right_code.begin(), right_code.end());

  } else if (hit_right) {
    //Huffman encoding compresses 01 into 0
    code.push_back(false);

    std::vector<bool> right_code = getHitmapCode(right_hitmap);
    code.insert(code.end(), right_code.begin(), right_code.end());

  } else if (hit_left) {
    code.push_back(true);
    code.push_back(false);

    std::vector<bool> left_code = getHitmapCode(left_hitmap);
    code.insert(code.end(), left_code.begin(), left_code.end());
  }

  return code;
}

//Returns the bit code associated with the Phase2ITQCore
std::vector<bool> Phase2ITQCore::encodeQCore(bool is_new_col) {
  std::vector<bool> code = {};

  if (is_new_col) {
    std::vector<bool> col_code = intToBinary(ccol_, 6);
    code.insert(code.end(), col_code.begin(), col_code.end());
  }

  code.push_back(islast_);
  code.push_back(isneighbour_);

  if (!isneighbour_) {
    std::vector<bool> row_code = intToBinary(qcrow_, 8);
    code.insert(code.end(), row_code.begin(), row_code.end());
  }

  std::vector<bool> hitmap_code = getHitmapCode(getHitmap());
  code.insert(code.end(), hitmap_code.begin(), hitmap_code.end());

  for (auto adc : adcs_) {
    std::vector<bool> adc_code = intToBinary(adc, 4);
    code.insert(code.end(), adc_code.begin(), adc_code.end());
  }

  return code;
}
