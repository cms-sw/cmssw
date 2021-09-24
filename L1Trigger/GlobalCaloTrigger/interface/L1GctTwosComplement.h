#ifndef L1GCTETTYPES_H
#define L1GCTETTYPES_H

#include <ostream>
#include <cstdint>

/*!
 * \class L1GctTwosComplement
 * \brief Definition of signed integer types with overflow
 *
 * This file defines the template class L1GctTwosComplement. It is used
 * to store energy values that are represented in a given number of bits
 * in hardware. The type has a built-in overFlow that is set if the value
 * to be represented is outside the allowed range for that number of bits.
 * This type represents signed integers; unsigned integers are represented
 * by L1GctUnsignedInt. Functions are defined to add two values,
 * and to copy data into a different number of bits.
 *
 * this header file contains method definitions because these are template classes
 * see http://www.parashift.com/c++-faq-lite/templates.html#faq-35.12
 *
 * \author Jim Brooke & Greg Heath
 * \date May 2006
 * 
 */

template <int nBits>
class L1GctTwosComplement {
public:
  /// Construct a signed integer with initial value zero
  L1GctTwosComplement();
  /// Construct a signed integer from raw data, checking for overFlow
  L1GctTwosComplement(uint32_t raw);
  /// Construct a signed integer, checking for overflow
  L1GctTwosComplement(int value);
  /// Destructor
  ~L1GctTwosComplement() {}

  /// Copy contructor to move data between representations with different numbers of bits
  template <int mBits>
  L1GctTwosComplement(const L1GctTwosComplement<mBits>& rhs);

  /// reset value and overflow to zero
  void reset();

  /// set the raw data
  void setRaw(uint32_t raw);

  /// set value from signed int
  void setValue(int value);

  /// set the overflow bit
  void setOverFlow(bool oflow) { m_overFlow = oflow; }

  /// access raw data
  uint32_t raw() const { return m_data; }

  /// access value as signed int
  int value() const;

  /// access overflow
  bool overFlow() const { return m_overFlow; }

  /// return number of bits
  int size() const { return m_nBits; }

  /// add two numbers of the same size
  L1GctTwosComplement operator+(const L1GctTwosComplement& rhs) const;

  /// overload unary - (negation) operator
  L1GctTwosComplement operator-() const;

  /// overload = operator
  L1GctTwosComplement& operator=(int value);

private:
  // number of bits (for overflow checking)
  int m_nBits;

  // the raw data
  uint32_t m_data;

  // the overflow bit
  bool m_overFlow;

  static const int MAX_NBITS = 24;
  static const int MAX_VALUE = 1 << (MAX_NBITS - 1);

  // PRIVATE MEMBER FUNCTION
  // function to check overflow
  void checkOverFlow(uint32_t rawValue, uint32_t& maskValue, bool& overFlow);
};

template <int nBits>
std::ostream& operator<<(std::ostream& s, const L1GctTwosComplement<nBits>& data);

// construct with # bits and set to zero
template <int nBits>
L1GctTwosComplement<nBits>::L1GctTwosComplement() {
  m_nBits = nBits > 0 && nBits < MAX_NBITS ? nBits : 16;
  this->reset();
}

// construct from # bits and raw data
template <int nBits>
L1GctTwosComplement<nBits>::L1GctTwosComplement(uint32_t raw) {
  m_nBits = nBits > 0 && nBits < MAX_NBITS ? nBits : 16;
  m_overFlow = false;
  this->setRaw(raw);
}

// construct from # bits and value
template <int nBits>
L1GctTwosComplement<nBits>::L1GctTwosComplement(int value) {
  m_nBits = nBits > 0 && nBits < MAX_NBITS ? nBits : 16;
  m_overFlow = false;
  this->setValue(value);
}

// copy contructor to move data between
// representations with different numbers of bits
template <int nBits>
template <int mBits>
L1GctTwosComplement<nBits>::L1GctTwosComplement(const L1GctTwosComplement<mBits>& rhs) {
  m_nBits = nBits > 0 && nBits < MAX_NBITS ? nBits : 16;
  this->setRaw(rhs.raw());
  this->setOverFlow(this->overFlow() || rhs.overFlow());
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
  if (chkValue < -MAX_VALUE) {
    chkValue = -MAX_VALUE;
    m_overFlow = true;
  }
  if (chkValue >= MAX_VALUE) {
    chkValue = MAX_VALUE - 1;
    m_overFlow = true;
  }

  // Transform negative values to large positive values
  posValue = chkValue < 0 ? chkValue + (1 << MAX_NBITS) : chkValue;
  raw = static_cast<uint32_t>(posValue);

  // Use the setRaw method to check overflow for our given size nBits
  this->setRaw(raw);
}

// return value as int
template <int nBits>
int L1GctTwosComplement<nBits>::value() const {
  int value, result;
  int maxValueInNbits;
  maxValueInNbits = 1 << (m_nBits - 1);
  value = static_cast<int>(m_data);
  result = value < maxValueInNbits ? value : value - (1 << MAX_NBITS);
  return result;
}

// add two numbers
template <int nBits>
L1GctTwosComplement<nBits> L1GctTwosComplement<nBits>::operator+(const L1GctTwosComplement<nBits>& rhs) const {
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

// overload unary - (negation) operator
template <int nBits>
L1GctTwosComplement<nBits> L1GctTwosComplement<nBits>::operator-() const {
  L1GctTwosComplement<nBits> temp;
  temp.setValue(-this->value());
  temp.setOverFlow(temp.overFlow() || this->overFlow());
  return temp;
}

// overload assignment by int
template <int nBits>
L1GctTwosComplement<nBits>& L1GctTwosComplement<nBits>::operator=(int value) {
  this->setValue(value);
  return *this;
}

// Here's the check overflow function
template <int nBits>
void L1GctTwosComplement<nBits>::checkOverFlow(uint32_t rawValue, uint32_t& maskValue, bool& overFlow) {
  uint32_t signBit = 1 << (m_nBits - 1);
  uint32_t signExtendBits = (static_cast<uint32_t>(MAX_VALUE) - signBit) << 1;
  // Consider and return only MAX_NBITS least significant bits
  uint32_t mskRawValue = rawValue & ((1 << MAX_NBITS) - 1);
  uint32_t value;
  bool ofl;

  if ((mskRawValue & signBit) == 0) {
    value = mskRawValue & ~signExtendBits;
  } else {
    value = mskRawValue | signExtendBits;
  }
  ofl = value != mskRawValue;

  maskValue = value;
  overFlow = ofl;
}

// overload ostream<<
template <int nBits>
std::ostream& operator<<(std::ostream& s, const L1GctTwosComplement<nBits>& data) {
  s << "L1GctTwosComplement<" << data.size() << "> raw : " << data.raw() << ", "
    << "value : " << data.value();
  if (data.overFlow()) {
    s << " Overflow set! ";
  }

  return s;
}

#endif
