

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEtTypes.h"

#include <iostream>

int main() {

  // test construction and set value
  L1GctTwosComplement a(12);
  a.setValue(1);
  std::cout << a << std::endl;

  // test operator=
  L1GctTwosComplement b(12);
  b = -3;
  std::cout << b << std::endl;

  // test addition
  L1GctTwosComplement c(13);
  c = a + b;
  std::cout << c << std::endl;

  // test addition with wrong number of bits
  L1GctTwosComplement d(12);
  d = a + b;
  std::cout << d << std::endl;


}
