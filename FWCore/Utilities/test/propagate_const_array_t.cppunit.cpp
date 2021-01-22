/*----------------------------------------------------------------------

Test program for edm::propagate_const_array class.

 ----------------------------------------------------------------------*/

#include <cppunit/extensions/HelperMacros.h>
#include <memory>
#include "FWCore/Utilities/interface/get_underlying_safe.h"
#include "FWCore/Utilities/interface/propagate_const_array.h"

class test_propagate_const_array : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(test_propagate_const_array);

  CPPUNIT_TEST(test);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}

  void test();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(test_propagate_const_array);

namespace {
  class ConstChecker {
  public:
    int value() { return 0; }

    int const value() const { return 1; }
  };

  // used to check that edm::propagate_const_array<T[]> works with incomplete types
  class Incomplete;

}  // namespace

void test_propagate_const_array::test() {
  // test for edm::propagate_const_array<T[]>
  {
    ConstChecker checker[10];
    edm::propagate_const_array<ConstChecker[]> pChecker(checker);

    CPPUNIT_ASSERT(0 == pChecker[1].value());
    CPPUNIT_ASSERT(0 == pChecker.get()[1].value());
    CPPUNIT_ASSERT(0 == get_underlying_safe(pChecker)[1].value());

    const edm::propagate_const_array<ConstChecker[]> pConstChecker(checker);

    CPPUNIT_ASSERT(1 == pConstChecker[1].value());
    CPPUNIT_ASSERT(1 == pConstChecker.get()[1].value());
    CPPUNIT_ASSERT(1 == get_underlying_safe(pConstChecker)[1].value());
  }

  // test for edm::propagate_const_array<shared_ptr<T>>
  {
    // std::make_shared<T[]> requires C++20
    auto checker = std::shared_ptr<ConstChecker[]>(new ConstChecker[10]);
    edm::propagate_const_array<std::shared_ptr<ConstChecker[]>> pChecker(checker);

    CPPUNIT_ASSERT(0 == pChecker[1].value());
    CPPUNIT_ASSERT(0 == pChecker.get()[1].value());
    CPPUNIT_ASSERT(0 == get_underlying_safe(pChecker)[1].value());

    const edm::propagate_const_array<std::shared_ptr<ConstChecker[]>> pConstChecker(checker);

    CPPUNIT_ASSERT(1 == pConstChecker[1].value());
    CPPUNIT_ASSERT(1 == pConstChecker.get()[1].value());
    CPPUNIT_ASSERT(1 == get_underlying_safe(pConstChecker)[1].value());
  }

  // test for edm::propagate_const_array<unique_ptr<T>>
  {
    auto checker = std::make_unique<ConstChecker[]>(10);
    edm::propagate_const_array<std::unique_ptr<ConstChecker[]>> pChecker(std::move(checker));

    CPPUNIT_ASSERT(0 == pChecker[1].value());
    CPPUNIT_ASSERT(0 == pChecker.get()[1].value());
    CPPUNIT_ASSERT(0 == get_underlying_safe(pChecker)[1].value());

    const edm::propagate_const_array<std::unique_ptr<ConstChecker[]>> pConstChecker(
        std::make_unique<ConstChecker[]>(10));

    CPPUNIT_ASSERT(1 == pConstChecker[1].value());
    CPPUNIT_ASSERT(1 == pConstChecker.get()[1].value());
  }

  // test for edm::propagate_const_array<T[]> with incomplete types
  {
    edm::propagate_const_array<Incomplete[]> pIncomplete(nullptr);

    CPPUNIT_ASSERT(nullptr == get_underlying(pIncomplete));
  }
}
