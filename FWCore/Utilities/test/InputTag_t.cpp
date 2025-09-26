
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/TypeID.h"

#include "oneapi/tbb/parallel_for.h"
#include "oneapi/tbb/blocked_range.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace edm {
  class TestEDGetToken {
  public:
    static EDGetToken makeEDGetToken(unsigned int iValue) { return EDGetToken(iValue); }
  };

}  // namespace edm

// This is just for the test. Do not dereference the pointers.
// They points to nothing legal.

class InputTagModifier {
  edm::InputTag* inputTag;

public:
  InputTagModifier(edm::InputTag* i) : inputTag(i) {}

  void operator()(oneapi::tbb::blocked_range<int> const& r) const {
    for (int i = r.begin(); i != r.end(); i++) {
      for (unsigned int j = 0; j < 10000; ++j) {
        unsigned int index = inputTag->cachedToken().index();
        if (index != 5 && index != edm::EDGetToken().index())
          abort();
        // std::cout << "a\n";
      }
      for (unsigned int j = 0; j < 100; ++j) {
        inputTag->cacheToken(edm::TestEDGetToken::makeEDGetToken(5));
        // std::cout << "b\n";
      }
      for (unsigned int j = 0; j < 1000; ++j) {
        unsigned int index = inputTag->cachedToken().index();
        // if (index != 5) abort(); // This can fail!
        // std::cout << index << " c\n";
        if (index != 5 && index != edm::EDGetToken().index())
          abort();
      }
    }
  }
};

int main() {
  for (unsigned int k = 0; k < 500; ++k) {
    edm::InputTag tag21("a:b:c");
    InputTagModifier inputTagModifier(&tag21);
    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<int>(0, 20, 1), inputTagModifier);
  }
  return 0;
}
