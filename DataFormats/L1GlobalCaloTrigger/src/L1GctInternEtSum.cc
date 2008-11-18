#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctInternEtSum.h"



L1GctInternEtSum::L1GctInternEtSum() {

}


/// construct from individual quantities
L1GctInternEtSum::L1GctInternEtSum(uint16_t capBlock,
				   uint16_t capIndex,
				   int16_t bx,
				   uint32_t et,
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

L1GctInternEtSum L1GctInternEtSum::fromWheelHfRingSum(const uint16_t capBlock,
							     const uint16_t capIndex,
							     const int16_t bx,
							     const uint16_t data) {
  L1GctInternEtSum s;
  s.setEt(data & 0xff);
  s.setOflow(0);
  s.setType(wheel_hf_ring_et_sum);
  return s;
}

L1GctInternEtSum L1GctInternEtSum::fromWheelHfBitCount(const uint16_t capBlock,
							      const uint16_t capIndex,
							      const int16_t bx,
							      const uint16_t data) {
  L1GctInternEtSum s;
  s.setEt(data & 0x3f);
  s.setOflow(0);
  s.setType(wheel_hf_ring_bit_count);
  return s;
}


L1GctInternEtSum L1GctInternEtSum::fromJetTotEt(const uint16_t capBlock,
						       const uint16_t capIndex,
						       const int16_t bx,
						       const uint16_t data) {
  L1GctInternEtSum s;
  s.setEt(data & 0xfff);
  s.setOflow((data>>12)&0x1);
  s.setType(jet_tot_et);
  return s;
}


L1GctInternEtSum L1GctInternEtSum::fromJetMissEt(const uint16_t capBlock,
							const uint16_t capIndex,
							const int16_t bx,
							const uint32_t data) {
  L1GctInternEtSum s;
  s.setEt(data & 0xffff);
  s.setOflow((data>>17) & 0x1);
  s.setType(jet_miss_et);
  return s;
}


L1GctInternEtSum L1GctInternEtSum::fromTotalEt(const uint16_t capBlock,
						      const uint16_t capIndex,
						      const int16_t bx,
						      const uint32_t data) {
  L1GctInternEtSum s;
  s.setEt(data & 0xffff);
  s.setOflow((data>>17) & 0x1);
  s.setType(total_et);
  return s;
}


/// equality operator
bool L1GctInternEtSum::operator==(const L1GctInternEtSum& c) const {
  return ( data_ == c.raw() && bx_ == c.bx() );
}


/// set value
void L1GctInternEtSum::setValue(uint32_t val) {
  data_ &= 0x80000000;
  data_ |= val & 0x7ffffff;
}

/// set et
void L1GctInternEtSum::setEt(uint32_t et) {
  setValue(et);
}

/// set count
void L1GctInternEtSum::setCount(uint32_t count) {
  setValue(count);
}

/// set overflow bit
void L1GctInternEtSum::setOflow(uint8_t oflow) {
  data_ &= 0x7ffffff;
  data_ |= (oflow & 0x1)<<31;
}

/// Pretty-print operator for L1GctInternEtSum
std::ostream& operator<<(std::ostream& s, const L1GctInternEtSum& c) {
  s << "L1GctInternEtSum : ";
  s << " mag=" << c.et();
  if (c.oflow()) { s << "; overflow set"; }
  s << " cap block=" << c.capBlock(); 
  s << " index=" << c.capIndex(); 
  s << " BX=" << c.bx(); 
  return s;
}
