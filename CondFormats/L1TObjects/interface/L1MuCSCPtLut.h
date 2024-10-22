#ifndef L1TObjects_L1MuCSCPtLut_h
#define L1TObjects_L1MuCSCPtLut_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <string>
#include <cstring>

class CSCTFConfigProducer;

class L1MuCSCPtLut {
private:
  unsigned short pt_lut[1 << 21];
  friend class CSCTFConfigProducer;

public:
  void readFromDBS(std::string& ptLUT);

  unsigned short pt(unsigned long addr) const throw() {
    if (addr < (1 << 21))
      return pt_lut[(unsigned int)addr];
    else
      return 0;
  }

  const unsigned short* lut(void) const throw() { return pt_lut; }

  L1MuCSCPtLut& operator=(const L1MuCSCPtLut& lut) {
    memcpy(pt_lut, lut.pt_lut, sizeof(pt_lut));
    return *this;
  }

  L1MuCSCPtLut(void) { bzero(pt_lut, sizeof(pt_lut)); }
  L1MuCSCPtLut(const L1MuCSCPtLut& lut) { memcpy(pt_lut, lut.pt_lut, sizeof(pt_lut)); }
  ~L1MuCSCPtLut(void) {}

  COND_SERIALIZABLE;
};

#endif
