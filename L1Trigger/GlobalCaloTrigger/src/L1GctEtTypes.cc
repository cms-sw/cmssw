
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
L1GctTwosComplement
L1GctTwosComplement::operator+ (const L1GctTwosComplement &rhs) const {

  // temporary variable for storing the result (need to set its size)
  int size = this->nBits() + 1;
  L1GctTwosComplement temp(size);

  // do the addition here


  // return the temporary
  return temp;

}

// overload assignment by int
L1GctTwosComplement& L1GctTwosComplement::operator= (int value) {
  
  this->setValue(value);
  return *this;

}

// overload ostream<<
std::ostream& operator<<(std::ostream& s, const L1GctTwosComplement& data) {

  s << "L1GctTwosComplement raw : " << data.raw() << ", " << "value : " << data.value();
  if (data.overFlow()) { s << " Overflow set! "; }
  s << std::endl;
  return s;

}

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


/* unsigned integer representations */

L1GctUnsignedInt::L1GctUnsignedInt(int nBits) :
  m_nBits(nBits) 
{

}

L1GctUnsignedInt::L1GctUnsignedInt(int nBits, unsigned value) :
  m_nBits(nBits),
  m_value(value)
{

}

L1GctUnsignedInt::~L1GctUnsignedInt()
{

}

// set value, checking for overflow
void L1GctUnsignedInt::setValue(unsigned value)
{
  // check for overflow
  if (value >= (unsigned) 1<<(m_nBits-1) ) {
    m_overFlow = true;
  }

  // set value with bitmask
  m_value = value & (1<<(m_nBits-1) - 1);

}

// add two unsigneds
L1GctUnsignedInt
L1GctUnsignedInt::operator+ (const L1GctUnsignedInt &rhs) const {

  // temporary variable for storing the result (need to set its size)
  int size = this->nBits() + 1;
  L1GctUnsignedInt temp(size);

  // do the addition here
  // setValue() will automatically set the overflow if required
  temp.setValue( this->value() + rhs.value() );

  // return the temporary
  return temp;

}
