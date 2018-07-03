#include <functional>
#include "FWCore/Utilities/interface/EDGetToken.h"

namespace edm {
  class TestEDGetToken {
  public:
    template <typename... Args>
    static edm::EDGetToken makeToken( Args&&... iArgs) {
      return edm::EDGetToken(std::forward<Args>(iArgs)...);
    }
  
    template <typename T, typename... Args>
    static edm::EDGetTokenT<T> makeTokenT( Args&&... iArgs) {
      return edm::EDGetTokenT<T>(std::forward<Args>(iArgs)...);
    }
  };
}

#include <iostream>

int main() {

  edm::EDGetTokenT<int> token1 = edm::TestEDGetToken::makeTokenT<int>();
  if(!token1.isUninitialized() ||
     !(token1.index() == 0xFFFFFFFF)) {
    std::cout << "EDGetTokenT no argument constructor failed 1" << std::endl;
    abort();
  }

  edm::EDGetTokenT<int> token2 = edm::TestEDGetToken::makeTokenT<int>(11);
  if(token2.isUninitialized() ||
     !(token2.index() == 11)) {
    std::cout << "EDGetTokenT 1 argument constructor failed 2" << std::endl;
    abort();
  }

  edm::EDGetToken token10 = edm::TestEDGetToken::makeToken();
  if(!token10.isUninitialized() ||
     !(token10.index() == 0xFFFFFFFF)) {
    std::cout << "EDGetToken no argument constructor failed 10" << std::endl;
    abort();
  }

  edm::EDGetToken token11 = edm::TestEDGetToken::makeToken(100);
  if(token11.isUninitialized() ||
     !(token11.index() == 100)) {
    std::cout << "EDGetToken 1 argument constructor failed 11" << std::endl;
    abort();
  }

  edm::EDGetToken token12(token2);
  if(token12.isUninitialized() ||
     !(token12.index() == 11)) {
    std::cout << "EDGetToken 1 argument constructor failed 12" << std::endl;
    abort();
  }
}
