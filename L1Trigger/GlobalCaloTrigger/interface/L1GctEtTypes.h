#ifndef L1GCTETTYPES_H
#define L1GCTETTYPES_H

#include <boost/cstdint.hpp>
#include <ostream>


/* Signed integer representations */

// class to store a 2s complement number
// in the range -X to X-1
// including overflow

template <int nBits>
class L1GctTwosComplement {
 public:
  L1GctTwosComplement();
  L1GctTwosComplement(uint32_t raw);
  L1GctTwosComplement(int value);
  ~L1GctTwosComplement() { }

  // copy contructor to move data between
  // representations with different numbers of bits
  template <int mBits>
  L1GctTwosComplement(const L1GctTwosComplement<mBits>& rhs);

  // reset value and overflow to zero
  void reset();

  // set the raw data
  void setRaw(uint32_t raw);

  // set value from signed int
  void setValue(int value);

  // set the overflow bit
  void setOverFlow(bool oflow) { m_overFlow = oflow; }

  // access raw data
  uint32_t raw() const { return m_data; }

  // access value as signed int
  int value() const;

  // access overflow
  bool overFlow() const { return m_overFlow; }

  // return number of bits
  int size() const { return m_nBits; }

  // add two numbers of the same size
  L1GctTwosComplement operator+ (const L1GctTwosComplement &rhs) const;

  // overload = operator
  L1GctTwosComplement& operator= (int value);

 private:

  // number of bits (for overflow checking)
  int m_nBits;

  // the raw data
  uint32_t m_data;

  // the overflow bit
  bool m_overFlow;

  static const int MAX_NBITS = 24;
  static const int MAX_VALUE = 1<<(MAX_NBITS-1);

  // function to check overflow
  void checkOverFlow(uint32_t rawValue, uint32_t &maskValue, bool &overFlow);
};

template <int nBits>
std::ostream& operator<<(std::ostream& s, const L1GctTwosComplement<nBits>& data);


/* unsigned integer representations */

template <int nBits>
class L1GctUnsignedInt {

 public:

  L1GctUnsignedInt();
  L1GctUnsignedInt(unsigned value);
  ~L1GctUnsignedInt();

  // copy contructor to move data between
  // representations with different numbers of bits
  template <int mBits>
  L1GctUnsignedInt(const L1GctUnsignedInt<mBits>& rhs);

  // reset value and overflow to zero
  void reset() { m_value = static_cast<unsigned>(0); m_overFlow = false; }

  // Set value from unsigned
  void setValue(unsigned value);

  // set the overflow bit
  void setOverFlow(bool oflow) { m_overFlow = oflow; }

  // access value as unsigned
  unsigned value() const { return m_value; }

  // access overflow
  bool overFlow() const { return m_overFlow; }

  // return number of bits
  int size() const { return m_nBits; }

  // add two numbers
  L1GctUnsignedInt operator+ (const L1GctUnsignedInt &rhs) const;

  // overload = operator
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


/// Jet counts
template <int nBits>
class L1GctJetCount : public L1GctUnsignedInt<nBits> {

 public:

  L1GctJetCount();
  L1GctJetCount(unsigned value);
  ~L1GctJetCount();

  // copy contructor to move data between
  // representations with different numbers of bits
  template <int mBits>
  L1GctJetCount(const L1GctJetCount<mBits>& rhs);

  // Set value from unsigned
  void setValue(unsigned value);

  // set the overflow bit
  void setOverFlow(bool oflow);

  // since this is a counter, we want
  // increment operators
  L1GctJetCount& operator++ ();
  L1GctJetCount operator++ (int);

  // add two numbers
  L1GctJetCount operator+ (const L1GctJetCount &rhs) const;

  // overload = operator
  L1GctJetCount& operator= (int value);

};




/// Here are the typedef's for the data types used in the energy sum calculations

typedef L1GctTwosComplement<12> L1GctEtComponent;
typedef L1GctUnsignedInt<12>    L1GctScalarEtVal;
typedef L1GctUnsignedInt<7>     L1GctEtAngleBin;
typedef L1GctJetCount<5>        L1GctJcFinalType;
typedef L1GctJetCount<3>        L1GctJcWheelType;





#endif
