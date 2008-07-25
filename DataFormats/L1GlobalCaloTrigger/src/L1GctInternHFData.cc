
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctInternHFData.h"

L1GctInternHFData::L1GctInternHFData() { } 

/// destructor
L1GctInternHFData::~L1GctInternHFData() { }

L1GctInternHFData L1GctInternHFData::fromConcRingSums(const uint16_t capBlock,
						      const uint16_t capIndex,
						      const uint8_t bx,
						      const uint32_t data) {

}

L1GctInternHFData L1GctInternHFData::fromConcBitCounts(const uint16_t capBlock,
						       const uint16_t capIndex,
						       const uint8_t bx,
						       const uint32_t data) {

}

/// get the et sums
uint16_t L1GctInternHFData::et(unsigned const i) { }

/// get the counts
uint16_t L1GctInternHFData::count(unsigned const i) { } 


/// equality operator
bool L1GctInternHFData::operator==(const L1GctInternHFData& c) const { }


/// set the sum
void L1GctInternHFData::setEt(unsigned i, uint16_t et) { }

/// set the count
void L1GctInternHFData::setCount(unsigned i, uint16_t count) { }

std::ostream& operator<<(std::ostream& s, const L1GctInternHFData& cand);

