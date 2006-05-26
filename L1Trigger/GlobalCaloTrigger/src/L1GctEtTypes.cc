
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEtTypes.h"

// put code back here...

// construct with # bits and set to zero
template <int nBits>
L1GctTwosComplement<nBits>::L1GctTwosComplement() {
  m_nBits = nBits>0 && nBits<MAX_NBITS ? nBits : 16 ;
  this->reset();
}

// construct from # bits and raw data 
template <int nBits>
L1GctTwosComplement<nBits>::L1GctTwosComplement(uint32_t raw) {
  m_nBits = nBits>0 && nBits<MAX_NBITS ? nBits : 16 ;
  this->setRaw(raw);
}

// construct from # bits and value
template <int nBits>
L1GctTwosComplement<nBits>::L1GctTwosComplement(int value) {
  m_nBits = nBits>0 && nBits<MAX_NBITS ? nBits : 16 ;
  this->setValue(value);
}

// copy contructor to move data between
// representations with different numbers of bits
template <int nBits>
template <int mBits>
L1GctTwosComplement<nBits>::L1GctTwosComplement(const L1GctTwosComplement<mBits>& rhs) {
  m_nBits = nBits>0 && nBits<MAX_NBITS ? nBits : 16 ;
  this->setRaw( rhs.raw() );
  this->setOverFlow( this->overFlow() || rhs.overFlow() );
}

// reset value and overflow to zero
template <int nBits>
void L1GctTwosComplement<nBits>::reset() {
  m_data = static_cast<uint32_t>(0);
  m_overFlow = false;
}

// set value from uint32_t
template <int nBits>
void L1GctTwosComplement<nBits>::setRaw(uint32_t raw) {
  checkOverFlow(raw, m_data, m_overFlow);
}

// set value from int
template <int nBits>
void L1GctTwosComplement<nBits>::setValue(int value) {
  int chkValue, posValue;
  uint32_t raw;

  // Make sure we have an integer in the range MAX_NBITS
  chkValue = value;
  if (chkValue<-MAX_VALUE) { chkValue =  -MAX_VALUE; m_overFlow = true; }
  if (chkValue>=MAX_VALUE) { chkValue = MAX_VALUE-1; m_overFlow = true; }

  // Transform negative values to large positive values
  posValue = chkValue<0 ? chkValue + (1<<MAX_NBITS) : chkValue ;
  raw = static_cast<uint32_t>(posValue);

  // Use the setRaw method to check overflow for our given size nBits
  this->setRaw(raw);
}

// return value as int
template <int nBits>
int L1GctTwosComplement<nBits>::value() const {
  int value, result;
  int maxValueInNbits;
  maxValueInNbits = 1<<(m_nBits-1);
  value  = static_cast<int>(m_data);
  result = value < maxValueInNbits ? value : value - (1<<MAX_NBITS) ;
  return result;
}

// add two numbers
template <int nBits>
L1GctTwosComplement<nBits>
L1GctTwosComplement<nBits>::operator+ (const L1GctTwosComplement<nBits> &rhs) const {

  // temporary variable for storing the result (need to set its size)
  L1GctTwosComplement<nBits> temp;
  uint32_t sum;
  bool ofl;

  // do the addition here
  sum = this->raw() + rhs.raw();
  ofl = this->overFlow() || rhs.overFlow();

  //fill the temporary argument
  temp.setRaw(sum);
  temp.setOverFlow(temp.overFlow() || ofl);

  // return the temporary
  return temp;

}

// overload assignment by int
template <int nBits>
L1GctTwosComplement<nBits>& L1GctTwosComplement<nBits>::operator= (int value) {
  
  this->setValue(value);
  return *this;

}

// Here's the check overflow function
template <int nBits> 
void L1GctTwosComplement<nBits>::checkOverFlow(uint32_t rawValue, uint32_t &maskValue, bool &overFlow) {
  uint32_t signBit = 1<<(m_nBits-1);
  uint32_t signExtendBits = (static_cast<uint32_t>(MAX_VALUE)-signBit)<<1;
  uint32_t value;
  bool ofl;

  if ((rawValue&signBit)==0) {
    value = rawValue & ~signExtendBits;
  } else {
    value = rawValue | signExtendBits;
  }
  ofl = value != rawValue;

  maskValue = value;
  overFlow  = ofl;

}

/* unsigned integer representations */

template <int nBits>
L1GctUnsignedInt<nBits>::L1GctUnsignedInt() {

  m_nBits = nBits>0 && nBits<MAX_NBITS ? nBits : 16 ;
  this->reset();
}

template <int nBits>
L1GctUnsignedInt<nBits>::L1GctUnsignedInt(unsigned value) {

  m_nBits = nBits>0 && nBits<MAX_NBITS ? nBits : 16 ;
  this->setValue(value);
}

template <int nBits>
L1GctUnsignedInt<nBits>::~L1GctUnsignedInt()
{

}

// copy contructor to move data between
// representations with different numbers of bits
template <int nBits>
template <int mBits>
L1GctUnsignedInt<nBits>::L1GctUnsignedInt(const L1GctUnsignedInt<mBits>& rhs) {
  m_nBits = nBits>0 && nBits<MAX_NBITS ? nBits : 16 ;
  this->setValue( rhs.value() );
  this->setOverFlow( this->overFlow() || rhs.overFlow() );
}

// set value, checking for overflow
template <int nBits>
void L1GctUnsignedInt<nBits>::setValue(unsigned value)
{
  // check for overflow
  if (value >= (static_cast<unsigned>(1<<m_nBits)) ) {
    m_overFlow = true;
  }

  // set value with bitmask
  m_value = value & ((1<<m_nBits) - 1);

}

// add two unsigneds
template <int nBits>
L1GctUnsignedInt<nBits>
L1GctUnsignedInt<nBits>::operator+ (const L1GctUnsignedInt<nBits> &rhs) const {

  // temporary variable for storing the result (need to set its size)
  L1GctUnsignedInt<nBits> temp;

  unsigned sum;
  bool ofl;

  // do the addition here
  sum = this->value() + rhs.value();
  ofl = this->overFlow() || rhs.overFlow();

  //fill the temporary argument
  temp.setValue(sum);
  temp.setOverFlow(temp.overFlow() || ofl);

  // return the temporary
  return temp;

}

// overload assignment by int
template <int nBits>
L1GctUnsignedInt<nBits>& L1GctUnsignedInt<nBits>::operator= (int value) {
  
  this->setValue(value);
  return *this;

}

template <int nBits>
L1GctJetCount<nBits>::L1GctJetCount() : L1GctUnsignedInt<nBits>() {}

template <int nBits>
L1GctJetCount<nBits>::L1GctJetCount(unsigned value) : L1GctUnsignedInt<nBits>(value) {}

template <int nBits>
L1GctJetCount<nBits>::~L1GctJetCount() {}

// copy contructor to move data between
// representations with different numbers of bits
template <int nBits>
template <int mBits>
L1GctJetCount<nBits>::L1GctJetCount(const L1GctJetCount<mBits>& rhs) {
  m_nBits = nBits>0 && nBits<MAX_NBITS ? nBits : 16 ;
  this->setValue( rhs.value() );
  this->setOverFlow( this->overFlow() || rhs.overFlow() );
}

template <int nBits>
void L1GctJetCount<nBits>::setValue(unsigned value)
{
  // check for overflow
  if (value >= (static_cast<unsigned>((1<<m_nBits) - 1)) ) {
    m_overFlow = true;
    m_value = ((1<<m_nBits) - 1);
  } else {
    m_value = value;
  }

}

template <int nBits>
void L1GctJetCount<nBits>::setOverFlow(bool oflow)
{
  m_overFlow = oflow;
  if (oflow) { m_value = ((1<<m_nBits) - 1); }
}

// increment operators
template <int nBits>
L1GctJetCount<nBits>&
L1GctJetCount<nBits>::operator++ () {

  this->setValue(m_value+1);
  return *this;
}

template <int nBits>
L1GctJetCount<nBits>
L1GctJetCount<nBits>::operator++ (int) {

  L1GctJetCount<nBits> temp(m_value);
  temp.setOverFlow(m_overFlow);
  this->setValue(m_value+1);
  return temp;
}

// add two jet counts
template <int nBits>
L1GctJetCount<nBits>
L1GctJetCount<nBits>::operator+ (const L1GctJetCount<nBits> &rhs) const {

  // temporary variable for storing the result (need to set its size)
  L1GctJetCount<nBits> temp;

  unsigned sum;
  bool ofl;

  // do the addition here
  sum = this->value() + rhs.value();
  ofl = this->overFlow() || rhs.overFlow();

  //fill the temporary argument
  temp.setValue(sum);
  temp.setOverFlow(temp.overFlow() || ofl);

  // return the temporary
  return temp;

}

// overload assignment by int
template <int nBits>
L1GctJetCount<nBits>& L1GctJetCount<nBits>::operator= (int value) {
  
  this->setValue(value);
  return *this;

}

template <int nBits>
std::ostream& operator<<(std::ostream& s, const L1GctTwosComplement<nBits>& data);

template <int nBits>
std::ostream& operator<<(std::ostream& s, const L1GctUnsignedInt<nBits>& data);

// overload ostream<<
template <int nBits>
std::ostream& operator<<(std::ostream& s, const L1GctTwosComplement<nBits>& data) {

  s << "L1GctTwosComplement raw : " << data.raw() << ", " << "value : " << data.value();
  if (data.overFlow()) { s << " Overflow set! "; }
  s << std::endl;
  return s;

}

template <int nBits>
std::ostream& operator<<(std::ostream& s, const L1GctUnsignedInt<nBits>& data) {

  s << "L1GctUnsignedInt value : " << data.value();
  if (data.overFlow()) { s << " Overflow set! "; }
  s << std::endl;
  return s;

}

template <int nBits>
std::ostream& operator<<(std::ostream& s, const L1GctJetCount<nBits>& data) {

  s << "L1GctJetCount value : " << data.value();
  if (data.overFlow()) { s << " Overflow set! "; }
  s << std::endl;
  return s;

}
