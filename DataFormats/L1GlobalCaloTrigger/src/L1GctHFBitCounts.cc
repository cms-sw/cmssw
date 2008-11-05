
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctHFBitCounts.h"

/// set static consts
//static const unsigned L1GctHFBitCounts::N_COUNTS = 4;

/// default constructor (for vector initialisation etc.)
L1GctHFBitCounts::L1GctHFBitCounts() :
  capBlock_(0),
  capIndex_(0),
  bx_(0),
  data_(0) 
{ }


/// destructor
L1GctHFBitCounts::~L1GctHFBitCounts()
{ }

// named ctor for unpacker
L1GctHFBitCounts L1GctHFBitCounts::fromConcHFBitCounts(const uint16_t capBlock,
						       const uint16_t capIndex,
						       const int16_t bx,
						       const uint32_t data)
{
  L1GctHFBitCounts c;
  c.setCapBlock(capBlock);
  c.setCapIndex(capIndex);
  c.setBx(bx);
  c.setData(data&0xfff);
  return c;
}


// named ctor for GCT emulator
L1GctHFBitCounts L1GctHFBitCounts::fromGctEmulator(const int16_t bx,
						   const uint16_t bitCountPosEtaRing1,
						   const uint16_t bitCountNegEtaRing1,
						   const uint16_t bitCountPosEtaRing2,
						   const uint16_t bitCountNegEtaRing2)
{
  L1GctHFBitCounts c;
  c.setBx(bx);
  c.setBitCount(0, bitCountPosEtaRing1);
  c.setBitCount(1, bitCountNegEtaRing1);
  c.setBitCount(2, bitCountPosEtaRing2);
  c.setBitCount(3, bitCountNegEtaRing2);
  return c;  
}

 
/// get a bit count
///  index : sum
///    0   :  Ring 1 Positive Rapidity HF bit count
///    1   :  Ring 1 Negative Rapidity HF bit count
///    2   :  Ring 2 Positive Rapidity HF bit count
///    3   :  Ring 2 Negative Rapidity HF bit count
uint16_t L1GctHFBitCounts::bitCount(unsigned const i) const {
  return (data_>>(i*3)) & 0x7;
}


/// equality operator
bool L1GctHFBitCounts::operator==(const L1GctHFBitCounts& c) const {
  return (this->raw() == c.raw());
}


/// set a sum
void L1GctHFBitCounts::setBitCount(unsigned i, uint16_t c) {
  data_ &= ~(0x7<<(i*3));
  data_ |= (c&0x7)<<(i*3);
}


std::ostream& operator<<(std::ostream& s, const L1GctHFBitCounts& cand) 
{
  s << "L1GctHFBitCounts :";

  if (cand.empty()) {
    s << " empty";
  } else {    
    s << " ring1 eta+=" << cand.bitCount(0);
    s << " ring1 eta-=" << cand.bitCount(1);
    s << " ring2 eta+=" << cand.bitCount(2);
    s << " ring2 eta-=" << cand.bitCount(3);
    s << std::endl; 
  }

  s << std::hex << " cap block=" << cand.capBlock() << std::dec << " index=" << cand.capIndex() << " BX=" << cand.bx();

  return s;
}
