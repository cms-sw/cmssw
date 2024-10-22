
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctHtMiss.h"

L1GctHtMiss::L1GctHtMiss() : m_data(0), m_bx(0) {}

// The raw data is masked off so as only the MHT magnitude, overflow + phi bits are stored.
// This is because the raw data stream also contains a BC0 flag on bit 31, bit 15 is always
// set to 1, bits 14:13 are always '10', and bits 30:16 are always '101010...'.
// This data is masked off so as to match an L1GctHtMiss object constructed using
// the L1GctHtMiss(unsigned et, unsigned phi, bool oflow) constructor.
L1GctHtMiss::L1GctHtMiss(uint32_t rawData) : m_data(rawData & kRawCtorMask), m_bx(0) {}

L1GctHtMiss::L1GctHtMiss(uint32_t rawData, int16_t bx) : m_data(rawData & kRawCtorMask), m_bx(bx) {}

L1GctHtMiss::L1GctHtMiss(unsigned et, unsigned phi, bool oflow) : m_data(0), m_bx(0) {
  if ((et <= kHtMissMagMask) && (phi < kHtMissPhiNBins)) {
    m_data = (oflow ? kHtMissOFlowBit : 0) | ((et & kHtMissMagMask) << kHtMissMagShift) |
             ((phi & kHtMissPhiMask) << kHtMissPhiShift);
  } else {
    m_data = ((et & kHtMissMagMask) << kHtMissMagShift) | kHtMissOFlowBit;
  }
}

L1GctHtMiss::L1GctHtMiss(unsigned et, unsigned phi, bool oflow, int16_t bx) : m_data(0), m_bx(bx) {
  if ((et <= kHtMissMagMask) && (phi < kHtMissPhiNBins)) {
    m_data = (oflow ? kHtMissOFlowBit : 0) | ((et & kHtMissMagMask) << kHtMissMagShift) |
             ((phi & kHtMissPhiMask) << kHtMissPhiShift);
  } else {
    m_data = ((et & kHtMissMagMask) << kHtMissMagShift) | kHtMissOFlowBit;
  }
}

L1GctHtMiss::~L1GctHtMiss() {}

/// Pretty-print operator for L1GctHtMiss
std::ostream& operator<<(std::ostream& s, const L1GctHtMiss& c) {
  s << " L1GctHtMiss: ";
  s << " mag=" << c.et() << ", phi=" << c.phi();
  if (c.overFlow()) {
    s << "; overflow set";
  }
  return s;
}
