
#include "DataFormats/L1GlobalCaloTrigger/interface/L1EtSums.h"

L1EtTotal::L1EtTotal() : m_data(0) { }
L1EtTotal::L1EtTotal(uint16_t data) : m_data(data) { }
L1EtTotal::L1EtTotal(unsigned et, bool oflow) {
  m_data = (et & 0xfff) | (oflow ? 0x1000 : 0);
}
L1EtTotal::~L1EtTotal() { } 

L1EtHad::L1EtHad() : m_data(0) { } 
L1EtHad::L1EtHad(uint16_t data) : m_data(data) { }
L1EtHad::L1EtHad(unsigned et, bool oflow) {
  m_data = (et & 0xfff) | (oflow ? 0x1000 : 0);
}
L1EtHad::~L1EtHad() { } 

L1EtMiss::L1EtMiss() : m_data(0) { } 
L1EtMiss::L1EtMiss(uint32_t data) : m_data(data) { }
L1EtMiss::L1EtMiss(unsigned et, unsigned phi, bool oflow) {
  m_data = et | (oflow ? 0x1000 : 0) | ((phi & 0x7f)<<13) ;
}
L1EtMiss::~L1EtMiss() { } 
