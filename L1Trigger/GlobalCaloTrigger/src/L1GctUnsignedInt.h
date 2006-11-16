#ifndef L1GCTUNSIGNEDINT_H
#define L1GCTUNSIGNEDINT_H

#include <boost/cstdint.hpp>
#include <ostream>

/* unsigned integer representations */


template <int nBits>
class L1GctUnsignedInt {

 public:

  /// Construct an unsigned integer with initial value zero
  L1GctUnsignedInt();
  /// Construct an unsigned integer and check for overFlow
  L1GctUnsignedInt(unsigned value);
  /// Destructor
  ~L1GctUnsignedInt();

  /// Copy contructor to move data between representations with different numbers of bits
  template <int mBits> L1GctUnsignedInt(const L1GctUnsignedInt<mBits>& rhs);

  /// reset value and overflow to zero
  void reset() { m_value = static_cast<unsigned>(0); m_overFlow = false; }

  /// Set value from unsigned
  void setValue(unsigned value);

  /// set the overflow bit
  void setOverFlow(bool oflow) { m_overFlow = oflow; }

  /// access value as unsigned
  unsigned value() const { return m_value; }

  /// access overflow
  bool overFlow() const { return m_overFlow; }

  /// return number of bits
  int size() const { return m_nBits; }

  /// add two numbers
  L1GctUnsignedInt operator+ (const L1GctUnsignedInt &rhs) const;

  /// overload = operator
  L1GctUnsignedInt& operator= (int value);

 protected:

  // number of bits
  int m_nBits;

  // value
  unsigned m_value;

  // overflow
  bool m_overFlow;

  static const int MAX_NBITS = 24;

};

template <int nBits>
std::ostream& operator<<(std::ostream& s, const L1GctUnsignedInt<nBits>& data);

template <int nBits>
L1GctUnsignedInt<nBits>::L1GctUnsignedInt() {

  m_nBits = nBits>0 && nBits<MAX_NBITS ? nBits : 16 ;
  this->reset();
}

template <int nBits>
L1GctUnsignedInt<nBits>::L1GctUnsignedInt(unsigned value) {

  m_nBits = nBits>0 && nBits<MAX_NBITS ? nBits : 16 ;
  m_overFlow = false;
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


// removed typedefs for slc4 compilation

/// typedef for the data type used for Ex and Ey in the energy sum calculations
//typedef L1GctTwosComplement<12> L1GctEtComponent;
/// typedef for the data type used for Et and Ht, and missing Et magnitude, in the energy sum calculations
//typedef L1GctUnsignedInt<12>    L1GctScalarEtVal;
/// typedef for the data type used for missing Et phi bin in the energy sum calculations
//typedef L1GctUnsignedInt<7>     L1GctEtAngleBin;



#endif

