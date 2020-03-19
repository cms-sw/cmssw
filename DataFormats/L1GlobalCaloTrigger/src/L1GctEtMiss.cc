
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtMiss.h"

L1GctEtMiss::L1GctEtMiss() : m_data(0), m_bx(0) {}

// The raw data is masked off so as only the MET magnitude, overflow + phi bits are stored.
// This is because the raw data stream also contains a BC0 flag on bit 31, and bit 15 is always
// set to 1.  This data is masked off so as to match an L1GctEtMiss object constructed using
// the L1GctEtMiss(unsigned et, unsigned phi, bool oflow) constructor.
L1GctEtMiss::L1GctEtMiss(uint32_t rawData) : m_data(rawData & kRawCtorMask), m_bx(0) {}

L1GctEtMiss::L1GctEtMiss(uint32_t rawData, int16_t bx) : m_data(rawData & kRawCtorMask), m_bx(bx) {}

L1GctEtMiss::L1GctEtMiss(unsigned et, unsigned phi, bool oflow) : m_data(0), m_bx(0) {
  if ((et <= kEtMissMaxValue) && (phi < kEtMissPhiNBins)) {
    m_data = et | (oflow ? kEtMissOFlowBit : 0) | ((phi & kETMissPhiMask) << kEtMissPhiShift);
  } else {
    m_data = (et & kEtMissMaxValue) | kEtMissOFlowBit;
  }
}

L1GctEtMiss::L1GctEtMiss(unsigned et, unsigned phi, bool oflow, int16_t bx) : m_data(0), m_bx(bx) {
  if ((et <= kEtMissMaxValue) && (phi < kEtMissPhiNBins)) {
    m_data = et | (oflow ? kEtMissOFlowBit : 0) | ((phi & kETMissPhiMask) << kEtMissPhiShift);
  } else {
    m_data = (et & kEtMissMaxValue) | kEtMissOFlowBit;
  }
}

L1GctEtMiss::~L1GctEtMiss() {}

/// Pretty-print operator for L1GctEtMiss
std::ostream& operator<<(std::ostream& s, const L1GctEtMiss& c) {
  s << " L1GctEtMiss: ";
  s << " mag=" << c.et() << ", phi=" << c.phi();
  if (c.overFlow()) {
    s << "; overflow set";
  }
  return s;
}
