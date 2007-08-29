
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"

L1GctEtTotal::L1GctEtTotal() : m_data(0) { }
L1GctEtTotal::L1GctEtTotal(uint16_t data) : m_data(data) { }
L1GctEtTotal::L1GctEtTotal(unsigned et, bool oflow) {
  m_data = (et & 0xfff) | (oflow ? 0x1000 : 0);
}
L1GctEtTotal::~L1GctEtTotal() { } 

L1GctEtHad::L1GctEtHad() : m_data(0) { } 
L1GctEtHad::L1GctEtHad(uint16_t data) : m_data(data) { }
L1GctEtHad::L1GctEtHad(unsigned et, bool oflow) {
  m_data = (et & 0xfff) | (oflow ? 0x1000 : 0);
}
L1GctEtHad::~L1GctEtHad() { } 

L1GctEtMiss::L1GctEtMiss() : m_data(0) { } 
L1GctEtMiss::L1GctEtMiss(uint32_t data) : m_data(data) { }
L1GctEtMiss::L1GctEtMiss(unsigned et, unsigned phi, bool oflow) {
  m_data = et | (oflow ? 0x1000 : 0) | ((phi & 0x7f)<<13) ;
}
L1GctEtMiss::~L1GctEtMiss() { } 

/// Pretty-print operator for L1GctEtTotal
std::ostream& operator<<(std::ostream& s, const L1GctEtTotal& c) {
  s << " L1GctEtTotal: ";
  s << " et=" << c.et();
  if (c.overFlow()) { s << "; overflow set"; }
  return s;
}

/// Pretty-print operator for L1GctEtHad
std::ostream& operator<<(std::ostream& s, const L1GctEtHad& c) {
  s << " L1GctEtHad: ";
  s << " ht=" << c.et();
  if (c.overFlow()) { s << "; overflow set"; }
  return s;
}

/// Pretty-print operator for L1GctEtMiss
std::ostream& operator<<(std::ostream& s, const L1GctEtMiss& c) {
  s << " L1GctEtMiss: ";
  s << " mag=" << c.et() << ", phi=" << c.phi();
  if (c.overFlow()) { s << "; overflow set"; }
  return s;
}

