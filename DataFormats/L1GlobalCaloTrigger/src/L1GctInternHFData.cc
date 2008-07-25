
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctInternHFData.h"

L1GctInternHFData::L1GctInternHFData() { } 

/// destructor
L1GctInternHFData::~L1GctInternHFData() { }

L1GctInternHFData L1GctInternHFData::fromConcRingSums(const uint16_t capBlock,
						      const uint16_t capIndex,
						      const uint8_t bx,
						      const uint32_t data) {
  L1GctInternHFData d;
  d.setType(conc_hf_ring_et_sums);
  d.setCapIndex(capIndex);
  d.setCapBlock(capBlock);
  d.setData(data);
}

L1GctInternHFData L1GctInternHFData::fromConcBitCounts(const uint16_t capBlock,
						       const uint16_t capIndex,
						       const uint8_t bx,
						       const uint32_t data) {
  L1GctInternHFData d;
  d.setType(conc_hf_bit_counts);
  d.setCapIndex(capIndex);
  d.setCapBlock(capBlock);
  for (unsigned i=0; i<4; ++i) {
    d.setCount(i, (data>>(6*i))&0x3f);
  }
}

/// get the et sums
uint16_t L1GctInternHFData::et(unsigned const i) {
  return (data_<<(i*8)) & 0xff;
}

/// get the counts
uint16_t L1GctInternHFData::count(unsigned const i) {
  return (data_<<(i*8)) & 0xff;
} 


/// equality operator
bool L1GctInternHFData::operator==(const L1GctInternHFData& c) const { }


/// set the sum
void L1GctInternHFData::setEt(unsigned i, uint16_t et) {
  data_ &= ~(0xff<<(i*8));
  data_ |= (et&0xff)<<(i*8);
}

/// set the count
void L1GctInternHFData::setCount(unsigned i, uint16_t count) {
  data_ &= ~(0xff<<(i*8));
  data_ |= (count&0xff)<<(i*8);
}


std::ostream& operator<<(std::ostream& s, const L1GctInternHFData& cand);

