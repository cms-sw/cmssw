#ifndef L1GCTJETCOUNT_H
#define L1GCTJETCOUNT_H

#include "L1Trigger/GlobalCaloTrigger/src/L1GctUnsignedInt.h"
#include "L1Trigger/GlobalCaloTrigger/src/L1GctTwosComplement.h"

#include <ostream>

/*!
 * \class L1GctJetCount
 * \brief Definition of unsigned integer types with increment and overflow
 *
 * This file defines the template class L1GctJetCount. It is used to store
 * energy values that are represented in a given number of bits in hardware.
 * The value is set to the maximum (all bits set to '1') to represent an overFlow
 * condition, if the number to be represented is outside the allowed range
 * for that number of bits. The counters are unsigned integers.
 * Functions are defined to increment the counter, to add two values,
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
class L1GctJetCount : public L1GctUnsignedInt<nBits> {
public:
  /// Construct a counter and initialise its value to zero
  L1GctJetCount();
  /// Construct a counter, checking for overFlow
  L1GctJetCount(unsigned value);
  /// Destructor
  ~L1GctJetCount();

  /// Copy contructor to move data between representations with different numbers of bits
  template <int mBits>
  L1GctJetCount(const L1GctJetCount<mBits>& rhs);

  /// Set value from unsigned
  void setValue(unsigned value);

  /// set the overflow bit
  void setOverFlow(bool oflow);

  /// Define increment operators, since this is a counter.
  L1GctJetCount& operator++();
  L1GctJetCount operator++(int);

  /// add two numbers
  L1GctJetCount operator+(const L1GctJetCount& rhs) const;

  /// overload = operator
  L1GctJetCount& operator=(int value);
};

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
  this->m_nBits = nBits > 0 && nBits < this->MAX_NBITS ? nBits : 16;
  this->setValue(rhs.value());
  this->setOverFlow(this->overFlow() || rhs.overFlow());
}

template <int nBits>
void L1GctJetCount<nBits>::setValue(unsigned value) {
  // check for overflow
  if (value >= (static_cast<unsigned>((1 << this->m_nBits) - 1))) {
    this->m_overFlow = true;
    this->m_value = ((1 << this->m_nBits) - 1);
  } else {
    this->m_value = value;
  }
}

template <int nBits>
void L1GctJetCount<nBits>::setOverFlow(bool oflow) {
  this->m_overFlow = oflow;
  if (oflow) {
    this->m_value = ((1 << this->m_nBits) - 1);
  }
}

// increment operators
template <int nBits>
L1GctJetCount<nBits>& L1GctJetCount<nBits>::operator++() {
  this->setValue(this->m_value + 1);
  return *this;
}

template <int nBits>
L1GctJetCount<nBits> L1GctJetCount<nBits>::operator++(int) {
  L1GctJetCount<nBits> temp(this->m_value);
  temp.setOverFlow(this->m_overFlow);
  this->setValue(this->m_value + 1);
  return temp;
}

// add two jet counts
template <int nBits>
L1GctJetCount<nBits> L1GctJetCount<nBits>::operator+(const L1GctJetCount<nBits>& rhs) const {
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
L1GctJetCount<nBits>& L1GctJetCount<nBits>::operator=(int value) {
  this->setValue(value);
  return *this;
}

// overload ostream<<
template <int nBits>
std::ostream& operator<<(std::ostream& s, const L1GctJetCount<nBits>& data) {
  s << "L1GctJetCount value : " << data.value();
  if (data.overFlow()) {
    s << " Overflow set! ";
  }

  return s;
}

// removed typedefs for slc4 compilation

/// typedef for the data type used for final output jet counts
//typedef L1GctJetCount<5>        L1GctJcFinalType;
/// typedef for the data type used for Wheel card jet counts
//typedef L1GctJetCount<3>        L1GctJcWheelType;

#endif
