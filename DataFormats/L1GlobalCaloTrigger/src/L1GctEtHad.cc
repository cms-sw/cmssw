
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtHad.h"

L1GctEtHad::L1GctEtHad() : m_data(0) { } 
L1GctEtHad::L1GctEtHad(uint16_t data) : m_data(data) { }
L1GctEtHad::L1GctEtHad(unsigned et, bool oflow) {
  m_data = (et & kEtHadMaxValue) | ((oflow || et>kEtHadMaxValue) ? kEtHadOFlowBit : 0);
}
L1GctEtHad::~L1GctEtHad() { } 

/// Pretty-print operator for L1GctEtHad
std::ostream& operator<<(std::ostream& s, const L1GctEtHad& c) {
  s << " L1GctEtHad: ";
  s << " ht=" << c.et();
  if (c.overFlow()) { s << "; overflow set"; }
  return s;
}
