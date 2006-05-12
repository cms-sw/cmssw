#ifndef L1GCTETTYPES_H
#define L1GCTETTYPES_H

#include <boost/cstdint.hpp>

// class to store a 2s complement number
// in the range -X to X-1
// including overflow
class L1GctTwosComplement {

 public:
  L1GctTwosComplement();
  L1GctTwosComplement(uint32_t raw);
  L1GctTwosComplement(int value);
  ~L1GctTwosComplement();

  // set the raw data
  void setRaw(uint32_t raw);

  // set value from signed int
  void setValue(int value);

  // set the overflow bit
  void setOverflow(bool oflow);

  // access raw data
  uint16_t raw();

  // access value as signed int
  int value();

  // access overflow
  bool overflow();

 private:

  // the raw data
  uint32_t data;

}



#endif
