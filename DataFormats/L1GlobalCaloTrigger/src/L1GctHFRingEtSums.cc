
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctHFRingEtSums.h"

/// set static consts
//static const unsigned L1GctHFRingEtSums::N_SUMS = 4;


/// default constructor (for vector initialisation etc.)
L1GctHFRingEtSums::L1GctHFRingEtSums() :
  capBlock_(0),
  capIndex_(0),
  bx_(0),
  data_(0) 
{ }


/// destructor
L1GctHFRingEtSums::~L1GctHFRingEtSums()
{ }


// named ctor for unpacker
L1GctHFRingEtSums L1GctHFRingEtSums::fromConcRingSums(const uint16_t capBlock,
						      const uint16_t capIndex,
						      const int16_t bx,
						      const uint32_t data)
{
  L1GctHFRingEtSums s;
  s.setCapBlock(capBlock);
  s.setCapIndex(capIndex);
  s.setBx(bx);
  s.setEtSum(0, (data>>12)&0x7 );
  s.setEtSum(1, (data>>16)&0x7 );
  s.setEtSum(2, (data>>19)&0x7 );
  s.setEtSum(3, (data>>22)&0x7 );
  return s;
}

// named ctor for GCT emulator
L1GctHFRingEtSums L1GctHFRingEtSums::fromGctEmulator(const int16_t bx,
						     const uint16_t etSumPosEtaRing1,
						     const uint16_t etSumNegEtaRing1,
						     const uint16_t etSumPosEtaRing2,
						     const uint16_t etSumNegEtaRing2)
{
  L1GctHFRingEtSums s;
  s.setBx(bx);
  s.setEtSum(0, etSumPosEtaRing1);
  s.setEtSum(1, etSumNegEtaRing1);
  s.setEtSum(2, etSumPosEtaRing2);
  s.setEtSum(3, etSumNegEtaRing2);
  return s;
}
  
/// get an Et sum
///  index : sum
///    0   :  Ring 1 Positive Rapidity HF Et sum
///    1   :  Ring 1 Negative Rapidity HF Et sum
///    2   :  Ring 2 Positive Rapidity HF Et sum
///    3   :  Ring 2 Negative Rapidity HF Et sum
uint16_t L1GctHFRingEtSums::etSum(unsigned const i) const {
  return (data_>>(i*3)) & 0x7;
}


/// equality operator
bool L1GctHFRingEtSums::operator==(const L1GctHFRingEtSums& c) const {
  return (this->raw() == c.raw());
}
  
/// set a sum
void L1GctHFRingEtSums::setEtSum(unsigned i, uint16_t et) {
  data_ &= ~(0x7<<(i*3));
  data_ |= (et&0x7)<<(i*3);
}

std::ostream& operator<<(std::ostream& s, const L1GctHFRingEtSums& cand) 
{
  s << "L1GctHFRingEtSums :";

  if (cand.empty()) {
    s << " empty";
  } else {    
    s << " ring1 eta+=" << cand.etSum(0);
    s << " ring1 eta-=" << cand.etSum(1);
    s << " ring2 eta+=" << cand.etSum(2);
    s << " ring2 eta-=" << cand.etSum(3);
    s << std::endl;
  }

  s << std::hex << " cap block=" << cand.capBlock() << std::dec << " index=" << cand.capIndex() << " BX=" << cand.bx();

  return s;
}
