
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/TypeID.h"

#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"

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

  void operator()(tbb::blocked_range<int> const& r) const {
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
  edm::InputTag tag1;
  if (tag1.label() != "" || tag1.instance() != "" || tag1.process() != "" || tag1.willSkipCurrentProcess()) {
    std::cout << "InputTag() failed" << std::endl;
    abort();
  }

  edm::InputTag tag2(std::string("a"), std::string("b"), std::string("c"));
  if (tag2.label() != "a" || tag2.instance() != "b" || tag2.process() != "c" || tag2.willSkipCurrentProcess()) {
    std::cout << "InputTag(string,string,string) failed" << std::endl;
    abort();
  }

  edm::InputTag tag3("d", "e", "f");
  if (tag3.label() != "d" || tag3.instance() != "e" || tag3.process() != "f" || tag3.willSkipCurrentProcess()) {
    std::cout << "InputTag(char*,char*,char*) failed" << std::endl;
    abort();
  }

  edm::InputTag tag4("g:h:i");
  if (tag4.label() != "g" || tag4.instance() != "h" || tag4.process() != "i" || tag4.willSkipCurrentProcess()) {
    std::cout << "InputTag(string) 1 failed" << std::endl;
    abort();
  }

  edm::InputTag tag5("g:h");
  if (tag5.label() != "g" || tag5.instance() != "h" || tag5.process() != "" || tag5.willSkipCurrentProcess()) {
    std::cout << "InputTag(string) 2 failed" << std::endl;
    abort();
  }

  edm::InputTag tag6("g");
  if (tag6.label() != "g" || tag6.instance() != "" || tag6.process() != "" || tag6.willSkipCurrentProcess()) {
    std::cout << "InputTag(string) 3 failed" << std::endl;
    abort();
  }

  edm::InputTag tag7(std::string("a"), std::string("b"), std::string("c"));
  edm::InputTag tag8(std::string("x"), std::string("b"), std::string("c"));
  if (!(tag2 == tag7) || tag4 == tag5 || tag5 == tag6 || tag7 == tag8) {
    std::cout << "InputTag::operator== failed" << std::endl;
    abort();
  }

  if (tag7.encode() != std::string("a:b:c") || tag5.encode() != std::string("g:h") ||
      tag6.encode() != std::string("g")) {
    std::cout << "InputTag::encode failed " << std::endl;
    abort();
  }

  edm::InputTag tag9(tag8);
  edm::InputTag tag11("a:b:c");
  edm::InputTag tag10(std::move(tag11));
  tag6 = tag10;
  tag5 = edm::InputTag("a:b:c");
  if (!(tag8 == tag9) || !(tag10 == tag7) || !(tag6 == tag10) || !(tag5 == tag10)) {
    std::cout << "InputTag::operator= or constructor, move or copy failed" << std::endl;
    abort();
  }

  edm::InputTag tag12("d", "e", "@skipCurrentProcess");
  if (tag12.label() != "d" || tag12.instance() != "e" || tag12.process() != "@skipCurrentProcess" ||
      !tag12.willSkipCurrentProcess()) {
    std::cout << "Test of tag12 failed" << std::endl;
    abort();
  }

  edm::InputTag tag12a(std::string("d"), std::string("e"), std::string("@skipCurrentProcess"));
  if (tag12a.label() != "d" || tag12a.instance() != "e" || tag12a.process() != "@skipCurrentProcess" ||
      !tag12a.willSkipCurrentProcess()) {
    std::cout << "Test of tag12a failed" << std::endl;
    abort();
  }

  edm::InputTag tag12b("d:e:@skipCurrentProcess");
  if (tag12b.label() != "d" || tag12b.instance() != "e" || tag12b.process() != "@skipCurrentProcess" ||
      !tag12b.willSkipCurrentProcess()) {
    std::cout << "Test of tag12b failed" << std::endl;
    abort();
  }

  edm::InputTag tag12c(tag12b);
  if (!tag12c.willSkipCurrentProcess()) {
    std::cout << "Test of tag12c failed" << std::endl;
    abort();
  }

  edm::InputTag tag12d;
  tag12d = tag12a;
  if (!tag12d.willSkipCurrentProcess()) {
    std::cout << "Test of tag12c failed" << std::endl;
    abort();
  }

  edm::InputTag tag13("d", "e", "@skipCurrentProcessx");
  if (tag13.willSkipCurrentProcess()) {
    std::cout << "Test of tag13 failed" << std::endl;
    abort();
  }

  edm::InputTag tag14("d", "e", "@skipCurrentProces");
  if (tag14.willSkipCurrentProcess()) {
    std::cout << "Test of tag14 failed" << std::endl;
    abort();
  }

  edm::InputTag tag15("d", "e", "@skipCurrentProcesx");
  if (tag15.willSkipCurrentProcess()) {
    std::cout << "Test of tag15 failed" << std::endl;
    abort();
  }

  unsigned int index = tag5.indexFor(testTypeID1, edm::InRun, reg1);
  if (index != edm::ProductResolverIndexInvalid) {
    std::cout << "InputTag::indexFor failed" << std::endl;
    abort();
  }

  tag5.tryToCacheIndex(5, testTypeID1, edm::InRun, reg1);
  tag5.tryToCacheIndex(6, testTypeID1, edm::InRun, reg1);

  index = tag5.indexFor(testTypeID1, edm::InRun, reg1);
  if (index != 5) {
    std::cout << "InputTag::indexFor call 2 failed" << std::endl;
    abort();
  }

  if (tag5.indexFor(testTypeID1, edm::InLumi, reg1) != edm::ProductResolverIndexInvalid ||
      tag5.indexFor(testTypeID1, edm::InRun, reg2) != edm::ProductResolverIndexInvalid ||
      tag5.indexFor(testTypeID2, edm::InRun, reg1) != edm::ProductResolverIndexInvalid) {
    std::cout << "InputTag::indexFor call 3 failed" << std::endl;
    abort();
  }

  for (unsigned int k = 0; k < 500; ++k) {
    edm::InputTag tag21("a:b:c");
    InputTagModifier inputTagModifier(&tag21);
    tbb::parallel_for(tbb::blocked_range<int>(0, 20, 1), inputTagModifier);
  }
  return 0;
}
