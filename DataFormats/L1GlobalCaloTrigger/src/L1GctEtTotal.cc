
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtTotal.h"

L1GctEtTotal::L1GctEtTotal() : m_data(0) { }
L1GctEtTotal::L1GctEtTotal(uint16_t data) : m_data(data) { }
L1GctEtTotal::L1GctEtTotal(unsigned et, bool oflow) {
  m_data = (et & kEtTotalMaxValue) | ((oflow || et>kEtTotalMaxValue) ? kEtTotalOFlowBit : 0);
}
L1GctEtTotal::~L1GctEtTotal() { } 

/// Pretty-print operator for L1GctEtTotal
std::ostream& operator<<(std::ostream& s, const L1GctEtTotal& c) {
  s << " L1GctEtTotal: ";
  s << " et=" << c.et();
  if (c.overFlow()) { s << "; overflow set"; }
  return s;
}
