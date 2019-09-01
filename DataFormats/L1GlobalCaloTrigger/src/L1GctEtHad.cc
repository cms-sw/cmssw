
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtHad.h"

L1GctEtHad::L1GctEtHad() : m_data(0), m_bx(0) {}

L1GctEtHad::L1GctEtHad(uint16_t rawData) : m_data(rawData & kRawCtorMask), m_bx(0) {}

L1GctEtHad::L1GctEtHad(uint16_t rawData, int16_t bx) : m_data(rawData & kRawCtorMask), m_bx(bx) {}

L1GctEtHad::L1GctEtHad(unsigned et, bool oflow) : m_data(0), m_bx(0) {
  m_data = (et & kEtHadMaxValue) | ((oflow || et > kEtHadMaxValue) ? kEtHadOFlowBit : 0);
}

L1GctEtHad::L1GctEtHad(unsigned et, bool oflow, int16_t bx) : m_data(0), m_bx(bx) {
  m_data = (et & kEtHadMaxValue) | ((oflow || et > kEtHadMaxValue) ? kEtHadOFlowBit : 0);
}

L1GctEtHad::~L1GctEtHad() {}

/// Pretty-print operator for L1GctEtHad
std::ostream& operator<<(std::ostream& s, const L1GctEtHad& c) {
  s << " L1GctEtHad: ";
  s << " ht=" << c.et();
  if (c.overFlow()) {
    s << "; overflow set";
  }
  return s;
}
