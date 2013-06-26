#include "DataFormats/Math/interface/SIMDVec.h"

#include<cmath>
#include<iostream>
#include<iomanip>
#include<typeinfo>
template<typename T>
                                                        
void print(T a, T b) {
  using namespace mathSSE;
  std::cout << typeid(T).name() << " "
            << a << " " << b << (samesign(a,b) ? " " : " not ") << "same sign" << std::endl;
}

int main() {
  using namespace mathSSE;
   // int mn = -0;
   // std::cout << mn << std::endl;
   // std::cout << std::hex << mn << std::endl;
  print(123,-902030);
  print(123LL,-902030LL);
  print(-123.f,123.e-4f);
  print(-123.,123.e-4);

  print(123, 902030);
  print(123LL,902030LL);
  print(123.f,123.e-4f);
  print(123.,123.e-4);

  print(-123,-902030);
  print(-123LL,-902030LL);
  print(-123.f,-123.e-4f);
  print(-123.,-123.e-4);

  //  int const mask= 0x80000000;
  // std::cout << mask << std::endl;
  // std::cout << std::hex << mask << std::endl;

   return 0;
}


