// have to do this evil in order to access constructors.
// This is acceptable only for white box tests like this.
#define private public
#include "FWCore/Utilities/interface/EDGetToken.h"
#undef private

#include <iostream>

int main() {

  edm::EDGetTokenT<int> token1;
  if(!token1.isUnitialized() ||
     !(token1.index() == 0xFFFFFFFF)) {
    std::cout << "EDGetTokenT no argument constructor failed 1" << std::endl;
    abort();
  }

  edm::EDGetTokenT<int> token2(11);
  if(token2.isUnitialized() ||
     !(token2.index() == 11)) {
    std::cout << "EDGetTokenT 1 argument constructor failed 2" << std::endl;
    abort();
  }

  edm::EDGetToken token10;
  if(!token10.isUnitialized() ||
     !(token10.index() == 0xFFFFFFFF)) {
    std::cout << "EDGetToken no argument constructor failed 10" << std::endl;
    abort();
  }

  edm::EDGetToken token11(100);
  if(token11.isUnitialized() ||
     !(token11.index() == 100)) {
    std::cout << "EDGetToken 1 argument constructor failed 11" << std::endl;
    abort();
  }

  edm::EDGetToken token12(token2);
  if(token12.isUnitialized() ||
     !(token12.index() == 11)) {
    std::cout << "EDGetToken 1 argument constructor failed 12" << std::endl;
    abort();
  }
}
