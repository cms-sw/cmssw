
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctInternHFData.h"

L1GctInternHFData::L1GctInternHFData() :
  type_(null),
  capBlock_(0),
  capIndex_(0),
  bx_(0),
  data_(0)
{ } 

/// destructor
L1GctInternHFData::~L1GctInternHFData() { }

L1GctInternHFData L1GctInternHFData::fromConcRingSums(const uint16_t capBlock,
						      const uint16_t capIndex,
						      const int16_t bx,
						      const uint32_t data) {
  L1GctInternHFData d;
  d.setType(conc_hf_ring_et_sums);
  d.setCapIndex(capIndex);
  d.setCapBlock(capBlock);
  d.setBx(bx);
  d.setData(data);
  return d;
}

L1GctInternHFData L1GctInternHFData::fromConcBitCounts(const uint16_t capBlock,
						       const uint16_t capIndex,
						       const int16_t bx,
						       const uint32_t data) {
  L1GctInternHFData d;
  d.setType(conc_hf_bit_counts);
  d.setCapIndex(capIndex);
  d.setCapBlock(capBlock);
  d.setBx(bx);
  for (unsigned i=0; i<4; ++i) {
    d.setCount(i, (data>>(6*i))&0x3f);
  }
  return d;
}

L1GctInternHFData L1GctInternHFData::fromWheelRingSums(const uint16_t capBlock,
                                                       const uint16_t capIndex,
                                                       const int16_t bx,
                                                       const uint32_t data) {
  L1GctInternHFData d;
  d.setType(wheel_hf_ring_et_sums);
  d.setCapIndex(capIndex);
  d.setCapBlock(capBlock);
  d.setBx(bx);
  d.setData(data & 0xff);
  return d;
}

L1GctInternHFData L1GctInternHFData::fromWheelBitCounts(const uint16_t capBlock,
                                                        const uint16_t capIndex,
                                                        const int16_t bx,
                                                        const uint32_t data) {
  L1GctInternHFData d;
  d.setType(wheel_hf_bit_counts);
  d.setCapIndex(capIndex);
  d.setCapBlock(capBlock);
  d.setBx(bx);
  d.setCount(0,data & 0x3f);
  return d;
}


// get value
uint16_t L1GctInternHFData::value(unsigned i) const {
  return (data_>>(i*8)) & 0xff;
}

/// get the et sums
uint16_t L1GctInternHFData::et(unsigned i) const {
  return value(i);
}

/// get the counts
uint16_t L1GctInternHFData::count(unsigned i) const {
  return value(i);
} 


/// equality operator
bool L1GctInternHFData::operator==(const L1GctInternHFData& c) const {
  return ( this->raw() == c.raw() );
}


// set value
void L1GctInternHFData::setValue(unsigned i, uint16_t val) {
  data_ &= ~(0xff<<(i*8));
  data_ |= (val&0xff)<<(i*8);
}

/// set the sum
void L1GctInternHFData::setEt(unsigned i, uint16_t et) {
  setValue(i, et);
}

/// set the count
void L1GctInternHFData::setCount(unsigned i, uint16_t count) {
  setValue(i, count);
}


std::ostream& operator<<(std::ostream& s, const L1GctInternHFData& cand)
{
  s << "L1GctInternHFData :";

  if (cand.empty()) {
    s << " empty";
  } else {      
    if (cand.type()==L1GctInternHFData::conc_hf_ring_et_sums){
      s << " type=conc_hf_ring_et_sums";
      s << " ring1 eta+=" << cand.et(0);
      s << " ring1 eta-=" << cand.et(1);
      s << " ring2 eta+=" << cand.et(2);
      s << " ring2 eta-=" << cand.et(3); 
   } else if (cand.type()==L1GctInternHFData::conc_hf_bit_counts){
      s << " type=conc_hf_bit_counts";
      s << " ring1 eta+=" << cand.count(0);
      s << " ring1 eta-=" << cand.count(1);
      s << " ring2 eta+=" << cand.count(2);
      s << " ring2 eta-=" << cand.count(3);
   } else if (cand.type()==L1GctInternHFData::wheel_hf_ring_et_sums){
     s << " type=conc_hf_ring_et_sums";
     s << " Et sum=" << cand.et(0);
   } else if (cand.type()==L1GctInternHFData::wheel_hf_bit_counts){
     s << " type=wheel_hf_bit_counts";
     s << " Bit count=" << cand.et(0);
   }
  }
  s << std::endl;
    
  s << std::hex << " cap block=" << cand.capBlock() << std::dec << " index=" << cand.capIndex() << " BX=" << cand.bx();

  return s;

}

