#include "L1Trigger/L1TMuonEndCap/interface/PtLUTReader.h"

#include <fstream>
#include <iostream>
#include <stdexcept>

#define PTLUT_SIZE (1<<30)

PtLUTReader::PtLUTReader() :
    ptlut_(),
    version_(4),
    ok_(false)
{

}

PtLUTReader::~PtLUTReader() {

}

void PtLUTReader::read(const std::string& lut_full_path) {
  if (ok_)  return;

  std::cout << "EMTF emulator: attempting to read pT LUT binary file from local area" << std::endl;
  std::cout << lut_full_path << std::endl;
  std::cout << "Non-standard operation; if it fails, now you know why" << std::endl;
  std::cout << "Be sure to check that the 'scale_pt' function still matches this LUT" << std::endl;
  std::cout << "Loading LUT, this might take a while..." << std::endl;

  std::ifstream infile(lut_full_path, std::ios::binary);
  if (!infile.good()) {
    char what[256];
    snprintf(what, sizeof(what), "Fail to open %s", lut_full_path.c_str());
    throw std::invalid_argument(what);
  }

  ptlut_.reserve(PTLUT_SIZE);

  typedef uint64_t full_word_t;
  full_word_t full_word;
  full_word_t sub_word[4] = {0, 0, 0, 0};

  while (infile.read(reinterpret_cast<char*>(&full_word), sizeof(full_word_t))) {
    sub_word[0] = (full_word>>0)      & 0x1FF;  // 9-bit
    sub_word[1] = (full_word>>9)      & 0x1FF;
    sub_word[2] = (full_word>>32)     & 0x1FF;
    sub_word[3] = (full_word>>(32+9)) & 0x1FF;

    ptlut_.push_back(sub_word[0]);
    ptlut_.push_back(sub_word[1]);
    ptlut_.push_back(sub_word[2]);
    ptlut_.push_back(sub_word[3]);
  }
  infile.close();

  if (ptlut_.size() != PTLUT_SIZE) {
    char what[256];
    snprintf(what, sizeof(what), "ptlut_.size() is %lu != %i", ptlut_.size(), PTLUT_SIZE);
    throw std::invalid_argument(what);
  }

  version_ = ptlut_.at(0);  // address 0 is the pT LUT version number
  ok_ = true;
  return;
}

PtLUTReader::content_t PtLUTReader::lookup(const address_t& address) const {
  return ptlut_.at(address);
}
