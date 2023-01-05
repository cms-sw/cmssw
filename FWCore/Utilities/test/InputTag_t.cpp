
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/TypeID.h"

#include "oneapi/tbb/parallel_for.h"
#include "oneapi/tbb/blocked_range.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace edm {
  class ProductRegistry;
}

// This is just for the test. Do not dereference the pointers.
// They points to nothing legal.
edm::ProductRegistry* reg1 = reinterpret_cast<edm::ProductRegistry*>(1);
edm::ProductRegistry* reg2 = reinterpret_cast<edm::ProductRegistry*>(2);

class TypeToTestInputTag1 {};
class TypeToTestInputTag2 {};
TypeToTestInputTag1 typeToTestInputTag1;
TypeToTestInputTag2 typeToTestInputTag2;

edm::TypeID testTypeID1(typeToTestInputTag1);
edm::TypeID testTypeID2(typeToTestInputTag2);

class InputTagModifier {
  edm::InputTag* inputTag;

public:
  InputTagModifier(edm::InputTag* i) : inputTag(i) {}

  void operator()(oneapi::tbb::blocked_range<int> const& r) const {
    for (int i = r.begin(); i != r.end(); i++) {
      for (unsigned int j = 0; j < 10000; ++j) {
        unsigned int index = inputTag->indexFor(testTypeID1, edm::InRun, reg1);
        if (index != 5 && index != edm::ProductResolverIndexInvalid)
          abort();
        // std::cout << "a\n";
      }
      for (unsigned int j = 0; j < 100; ++j) {
        inputTag->tryToCacheIndex(5, testTypeID1, edm::InRun, reg1);
        // std::cout << "b\n";
      }
      for (unsigned int j = 0; j < 1000; ++j) {
        unsigned int index = inputTag->indexFor(testTypeID1, edm::InRun, reg1);
        // if (index != 5) abort(); // This can fail!
        // std::cout << index << " c\n";
        if (index != 5 && index != edm::ProductResolverIndexInvalid)
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
