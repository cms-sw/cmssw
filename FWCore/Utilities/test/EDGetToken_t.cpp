// have to do this evil in order to access constructors.
// This is acceptable only for white box tests like this.
#define private public
#include "FWCore/Utilities/interface/EDGetToken.h"
#undef private

#include <iostream>

int main() {

  edm::EDGetTokenT<int> token1;
  if(!token1.isUnitialized() ||
     !(token1.index() == 0x7FFFFFFF) ||
     !token1.willSkipCurrentProcess()) {
    std::cout << "EDToken no argument constructor failed 1" << std::endl;
    abort();
  }
  edm::EDGetTokenT<int> token2(0x7FFFFFFF, true);
  if(!token2.isUnitialized() ||
     !(token2.index() == 0x7FFFFFFF) ||
     !token2.willSkipCurrentProcess()) {
    std::cout << "EDToken constructor failed 2" << std::endl;
    abort();
  }
  edm::EDGetTokenT<int> token3(0x7FFFFFFF, false);
  if(token3.isUnitialized() ||
     !(token3.index() == 0x7FFFFFFF) ||
     token3.willSkipCurrentProcess()) {
    std::cout << "EDToken constructor failed 3" << std::endl;
    abort();
  }
  edm::EDGetTokenT<int> token4(1, true);
  if(token4.isUnitialized() ||
     !(token4.index() == 1) ||
     !token4.willSkipCurrentProcess()) {
    std::cout << "EDToken constructor failed 4" << std::endl;
    abort();
  }
  edm::EDGetTokenT<int> token5(1, false);
  if(token5.isUnitialized() ||
     !(token5.index() == 1) ||
     token5.willSkipCurrentProcess()) {
    std::cout << "EDToken constructor failed 5" << std::endl;
    abort();
  }
  edm::EDGetToken token10;
  if(!token10.isUnitialized() ||
     !(token10.index() == 0x7FFFFFFF) ||
     !token10.willSkipCurrentProcess()) {
    std::cout << "EDToken no argument constructor failed 10" << std::endl;
    abort();
  }
  edm::EDGetToken token20(0x7FFFFFFF, true);
  if(!token20.isUnitialized() ||
     !(token20.index() == 0x7FFFFFFF) ||
     !token20.willSkipCurrentProcess()) {
    std::cout << "EDToken constructor failed 20" << std::endl;
    abort();
  }
  edm::EDGetToken token30(0x7FFFFFFF, false);
  if(token30.isUnitialized() ||
     !(token30.index() == 0x7FFFFFFF) ||
     token30.willSkipCurrentProcess()) {
    std::cout << "EDToken constructor failed 30" << std::endl;
    abort();
  }
  edm::EDGetToken token40(1, true);
  if(token40.isUnitialized() ||
     !(token40.index() == 1) ||
     !token40.willSkipCurrentProcess()) {
    std::cout << "EDToken constructor failed 40" << std::endl;
    abort();
  }
  edm::EDGetToken token50(1, false);
  if(token50.isUnitialized() ||
     !(token50.index() == 1) ||
     token50.willSkipCurrentProcess()) {
    std::cout << "EDToken constructor failed 50" << std::endl;
    abort();
  }
  edm::EDGetToken token60(token4);
  if(token60.isUnitialized() ||
     !(token60.index() == 1) ||
     !token60.willSkipCurrentProcess()) {
    std::cout << "EDToken constructor failed 60" << std::endl;
    abort();
  }
  edm::EDGetToken token70(token5);
  if(token70.isUnitialized() ||
     !(token70.index() == 1) ||
     token70.willSkipCurrentProcess()) {
    std::cout << "EDToken constructor failed 70" << std::endl;
    abort();
  }
}
