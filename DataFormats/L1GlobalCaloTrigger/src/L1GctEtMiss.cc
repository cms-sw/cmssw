
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtMiss.h"

L1GctEtMiss::L1GctEtMiss() : m_data(0) { } 
L1GctEtMiss::L1GctEtMiss(uint32_t data) : m_data(data) { }
L1GctEtMiss::L1GctEtMiss(unsigned et, unsigned phi, bool oflow) {
  if ((et <= kEtMissMaxValue) && (phi < kEtMissPhiNBins)) {
    m_data = et | (oflow ? kEtMissOFlowBit : 0) | ((phi & kETMissPhiMask)<<kEtMissPhiShift) ;
  } else {
    m_data = (et & kEtMissMaxValue) | kEtMissOFlowBit ;
  }
}
L1GctEtMiss::~L1GctEtMiss() { } 

/// Pretty-print operator for L1GctEtMiss
std::ostream& operator<<(std::ostream& s, const L1GctEtMiss& c) {
  s << " L1GctEtMiss: ";
  s << " mag=" << c.et() << ", phi=" << c.phi();
  if (c.overFlow()) { s << "; overflow set"; }
  return s;
}

