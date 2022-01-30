#include "L1Trigger/TrackFindingTracklet/interface/FPGAWord.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;
using namespace trklet;

FPGAWord::FPGAWord() {}

FPGAWord::FPGAWord(int value, int nbits, bool positive, int line, const char* file) {
  set(value, nbits, positive, line, file);
}

void FPGAWord::set(int value, int nbits, bool positive, int line, const char* file) {
  value_ = value;
  nbits_ = nbits;
  positive_ = positive;
  if (positive) {
    if (value < 0) {
      edm::LogProblem("Tracklet") << "FPGAWord got negative value:" << value << " (" << file << ":" << line << ")";
    }
    assert(value >= 0);
  }
  if (nbits >= 22) {
    edm::LogPrint("Tracklet") << "FPGAWord got too many bits:" << nbits << " (" << file << ":" << line << ")";
  }
  assert(nbits < 22);
  if (nbits <= 0) {
    edm::LogPrint("Tracklet") << "FPGAWord got too few bits:" << nbits << " (" << file << ":" << line << ")";
  }
  assert(nbits > 0);
  if (positive) {
    if (value >= (1 << nbits)) {
      if (file != nullptr) {
        edm::LogProblem("Tracklet") << "value too large:" << value << " " << (1 << nbits) << " (" << file << ":" << line
                                    << ")";
      }
    }
    assert(value < (1 << nbits));
  } else {
    if (value >= (1 << (nbits - 1))) {
      edm::LogProblem("Tracklet") << "value too large:" << value << " " << (1 << (nbits - 1)) - 1 << " (" << file << ":"
                                  << line << ")";
    }
    assert(value < (1 << (nbits - 1)));
    if (value < -(1 << (nbits - 1))) {
      edm::LogProblem("Tracklet") << "value too negative:" << value << " " << -(1 << (nbits - 1)) << " (" << file << ":"
                                  << line << ")";
    }
    assert(value >= -(1 << (nbits - 1)));
  }
}

std::string FPGAWord::str() const {
  const int nbit = nbits_;

  if (!(nbit > 0 && nbit < 22))
    edm::LogVerbatim("Tracklet") << "nbit: " << nbit;
  if (nbit == -1)
    return "?";
  if (nbit == 0)
    return "~";

  int valtmp = value_;
  string str = "";
  for (int i = 0; i < nbit; i++) {
    str = ((valtmp & 1) ? "1" : "0") + str;
    valtmp >>= 1;
  }

  return str;
}

unsigned int FPGAWord::bits(unsigned int lsb, unsigned int nbit) const {
  assert(lsb + nbit <= (unsigned int)nbits());
  return (value_ >> lsb) & ((1 << nbit) - 1);
}

bool FPGAWord::atExtreme() const {
  if (positive_) {
    return (value_ == 0) || (value_ == (1 << nbits_) - 1);
  }
  return ((value_ == (-(1 << (nbits_ - 1)))) || (value_ == ((1 << (nbits_ - 1)) - 1)));
}

bool FPGAWord::operator==(const FPGAWord& other) const {
  return (value_ == other.value_) && (nbits_ == other.nbits_) && (positive_ == other.positive_);
}
