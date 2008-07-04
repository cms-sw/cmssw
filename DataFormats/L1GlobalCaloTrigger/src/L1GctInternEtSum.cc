#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctInternEtSum.h"



L1GctInternEtSum::L1GctInternEtSum() {

}


/// construct from individual quantities
L1GctInternEtSum::L1GctInternEtSum(uint16_t capBlock,
				   uint16_t capIndex,
				   int16_t bx,
				   uint16_t et,
				   uint8_t oflow) :
  type_(null),
  capBlock_(capBlock),
  capIndex_(capIndex),
  bx_(bx),
  data_(0)
{
  this->setEt(et);
  this->setOflow(oflow);
}


/// destructor
L1GctInternEtSum::~L1GctInternEtSum() {

}


/// equality operator
bool L1GctInternEtSum::operator==(const L1GctInternEtSum& c) const {
  return ( data_ == c.raw() && bx_ == c.bx() );
}


/// set Et sum
void L1GctInternEtSum::setEt(uint16_t et) {
  data_ &= 0xfffe0000;
  data_ |= et & 0x1ffff;
}

/// set overflow bit
void L1GctInternEtSum::setOflow(uint8_t oflow) {
  data_ &= 0x1<<17;
  data_ |= (oflow & 0x1)<<17;
}
