
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtSums.h"

L1GctEtTotal::L1GctEtTotal() : theEtTotal(0) { } 
L1GctEtTotal::L1GctEtTotal(uint16_t data) : theEtTotal(data) { }
L1GctEtTotal::L1GctEtTotal(int et, bool oflow) {
  theEtTotal = et;
}
L1GctEtTotal::~L1GctEtTotal() { } 

L1GctEtHad::L1GctEtHad() : theEtHad(0) { } 
L1GctEtHad::L1GctEtHad(uint16_t data) : theEtHad(data) { }
L1GctEtHad::L1GctEtHad(int et, bool oflow) {
  theEtHad = et;
}
L1GctEtHad::~L1GctEtHad() { } 

L1GctEtMiss::L1GctEtMiss() : theEtMiss(0) { } 
L1GctEtMiss::L1GctEtMiss(uint32_t data) : theEtMiss(data) { }
L1GctEtMiss::L1GctEtMiss(int et, int phi, bool oflow) {
  theEtMiss = et + (phi & 0x7f)<<13;
}
L1GctEtMiss::~L1GctEtMiss() { } 
