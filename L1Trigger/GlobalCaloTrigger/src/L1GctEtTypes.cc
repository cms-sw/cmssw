
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEtTypes.h"

#include <iostream>

// construct with # bits and set to zero
L1GctTwosComplement::L1GctTwosComplement(int nBits) {

}

// construct from # bits and raw data 
L1GctTwosComplement::L1GctTwosComplement(int nBits, uint32_t raw) {

}

// construct from # bits and value
L1GctTwosComplement::L1GctTwosComplement(int nBits, int value) {

}

// destructor
L1GctTwosComplement::~L1GctTwosComplement() {

}

// set value from int
void L1GctTwosComplement::setValue(int value) {

}

// return value as int
int L1GctTwosComplement::value() const {

  return 0;

}

// add two numbers
L1GctTwosComplement operator+ (const L1GctTwosComplement lhs, const L1GctTwosComplement rhs) {

  // temporary variable for storing the result (need to set its size)
  L1GctTwosComplement temp(lhs.nBits() + 1);

  // do the addition here


  // return the temporary
  return temp;

}

std::ostream& operator<<(std::ostream& s, const L1GctTwosComplement data) {

  s << "2s comp : " << data.raw() << ", " << "value : " << data.value();
  if (data.overFlow()) { s << "Overflow set! "; }
  s << std::endl;
  return s;

}

/// fixed length types ///

// construct from raw data
L1GctEtComponent::L1GctEtComponent(uint32_t raw) :
  L1GctTwosComplement(N_BITS, raw) 
{

}

// construct from value
L1GctEtComponent::L1GctEtComponent(int value) : 
  L1GctTwosComplement(N_BITS, value)
{

}

L1GctEtComponent::~L1GctEtComponent() {
}
