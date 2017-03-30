#include "L1Trigger/L1TMuonEndCap/interface/PtLUTWriter.hh"

#include <fstream>
#include <iostream>
#include <stdexcept>

#define PTLUT_SIZE (1<<30)

PtLUTWriter::PtLUTWriter() :
    ptlut_(),
    version_(4),
    ok_(false)
{
  ptlut_.reserve(PTLUT_SIZE);
}

PtLUTWriter::~PtLUTWriter() {

}

void PtLUTWriter::write(const std::string& lut_full_path) const {
  //if (ok_)  return;

  std::cout << "Writing LUT, this might take a while..." << std::endl;

  std::ofstream outfile(lut_full_path, std::ios::binary);
  if (!outfile.good()) {
    char what[256];
    snprintf(what, sizeof(what), "Fail to open %s", lut_full_path.c_str());
    throw std::invalid_argument(what);
  }

  if (ptlut_.size() != PTLUT_SIZE) {
    char what[256];
    snprintf(what, sizeof(what), "ptlut_.size() is %lu != %i", ptlut_.size(), PTLUT_SIZE);
    throw std::invalid_argument(what);
  }

  ptlut_.at(0) = version_;  // address 0 is the pT LUT version number

  typedef uint64_t full_word_t;
  full_word_t full_word;
  full_word_t sub_word[4] = {0, 0, 0, 0};

  table_t::const_iterator ptlut_it  = ptlut_.begin();
  table_t::const_iterator ptlut_end = ptlut_.end();

  while (ptlut_it != ptlut_end) {
    sub_word[0] = *ptlut_it++;
    sub_word[1] = *ptlut_it++;
    sub_word[2] = *ptlut_it++;
    sub_word[3] = *ptlut_it++;

    full_word = 0;
    full_word |= ((sub_word[0] & 0x1FF) << 0);
    full_word |= ((sub_word[1] & 0x1FF) << 9);
    full_word |= ((sub_word[2] & 0x1FF) << 32);
    full_word |= ((sub_word[3] & 0x1FF) << (32+9));

    outfile.write(reinterpret_cast<char*>(&full_word), sizeof(full_word_t));
  }
  outfile.close();

  //ok_ = true;
  return;
}

void PtLUTWriter::push_back(const content_t& pt) {
  ptlut_.push_back(pt);
}
