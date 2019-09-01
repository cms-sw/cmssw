

#include "L1Trigger/GlobalCaloTrigger/src/L1GctTwosComplement.h"
#include "L1Trigger/GlobalCaloTrigger/src/L1GctUnsignedInt.h"
#include "L1Trigger/GlobalCaloTrigger/src/L1GctJetCount.h"

#include <iostream>

int main() {
  // test construction and set value
  L1GctTwosComplement<12> a;
  a.setValue(1500);
  std::cout << a << std::endl;

  // test operator=
  L1GctTwosComplement<12> b;
  b = 2000;
  std::cout << b << std::endl;

  // test addition
  L1GctTwosComplement<13> c;
  c = L1GctTwosComplement<13>(a) + L1GctTwosComplement<13>(b);
  std::cout << c << std::endl;

  c = L1GctTwosComplement<13>(a + b);
  std::cout << c << std::endl;

  // test addition with wrong number of bits
  L1GctTwosComplement<12> d;
  d = a + b;
  std::cout << d << std::endl;

  // test unsigned and jet count
  L1GctUnsignedInt<12> e;
  L1GctUnsignedInt<12> f;
  L1GctUnsignedInt<12> g;
  e = 13;
  f.setValue(97);
  g = e + f;
  std::cout << e << std::endl;
  std::cout << f << std::endl;
  std::cout << g << std::endl;

  L1GctJetCount<4> h;
  L1GctJetCount<4> i;
  L1GctJetCount<4> j;
  h = 13;
  i = h++;
  j = h + i;
  std::cout << h << std::endl;
  std::cout << i << std::endl;
  std::cout << j << std::endl;
}
