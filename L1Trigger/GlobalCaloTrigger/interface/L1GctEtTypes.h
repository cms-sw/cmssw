#ifndef L1GCTETTYPES_H
#define L1GCTETTYPES_H

#include <boost/cstdint.hpp>
#include <ostream>


/* Signed integer representations */

// class to store a 2s complement number
// in the range -X to X-1
// including overflow

class L1GctTwosComplement {
 public:
  L1GctTwosComplement(int nBits);
  L1GctTwosComplement(int nBits, uint32_t raw);
  L1GctTwosComplement(int nBits, int value);
  ~L1GctTwosComplement();

  // set the raw data
  void setRaw(uint32_t raw) { m_data = raw; }

  // set value from signed int
  void setValue(int value);

  // set the overflow bit
  void setOverFlow(bool oflow) { m_overFlow = oflow; }

  // access raw data
  uint16_t raw() const { return m_data; }

  // access value as signed int
  int value() const;

  // access overflow
  bool overFlow() const { return m_overFlow; }

  // return number of bits
  int nBits() const { return m_nBits; }


 private:

  // number of bits (for overflow checking)
  int m_nBits;

  // the raw data
  uint32_t m_data;

  // the overflow bit
  bool m_overFlow;

};

std::ostream& operator<<(std::ostream& s, const L1GctTwosComplement& data);

// specialisation for specific sizes
class L1GctEtComponent : public L1GctTwosComplement {

 public:

  // constructors do not have nBits arguments
  L1GctEtComponent(uint32_t raw);
  L1GctEtComponent(int data);
  ~L1GctEtComponent();

 private:

  static const int N_BITS = 12;

};



/* unsigned integer representations */

class L1GctUnsignedInt {

 public:

  L1GctUnsignedInt(int nBits);
  L1GctUnsignedInt(int nBits, unsigned value);
  ~L1GctUnsignedInt();

  // Set value from unsigned
  void setValue(unsigned value);

  // set the overflow bit
  void setOverFlow(bool oflow) { m_overFlow = oflow; }

  // access value as unsigned
  unsigned value() const { return m_value; }

  // access overflow
  bool overFlow() const { return m_overFlow; }

  // return number of bits
  int nBits() const { return m_nBits; }

 private:

  // number of bits
  int m_nBits;

  // value
  unsigned m_value;

  // overflow
  bool m_overFlow;

};

std::ostream& operator<<(std::ostream& s, const L1GctUnsignedInt& data);



#endif
