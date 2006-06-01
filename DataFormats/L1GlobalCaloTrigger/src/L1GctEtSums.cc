
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
