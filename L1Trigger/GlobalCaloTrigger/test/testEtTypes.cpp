

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEtTypes.h"

#include <iostream>

int main() {

  // test construction and set value
  L1GctTwosComplement<12> a;
  a.setValue(1);
  std::cout << a << std::endl;

  // test operator=
  L1GctTwosComplement<12> b;
  b = -3;
  std::cout << b << std::endl;

//   // test addition
//   L1GctTwosComplement<13> c;
//   c = a + b;
//   std::cout << c << std::endl;

  // test addition with wrong number of bits
  L1GctTwosComplement<12> d;
  d = a + b;
  std::cout << d << std::endl;


}
