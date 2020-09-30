#include "SimpleEDProductGetter.h"
#include "cppunit/extensions/HelperMacros.h"
#include "DataFormats/Common/interface/refToElementID.h"
#include "DataFormats/Common/interface/EDProductGetter.h"
#include <iostream>

namespace testreftoelementid {
  struct Base {
    virtual ~Base() {}
    virtual int val() const = 0;
  };

  struct Inherit1 : public Base {
    virtual int val() const { return 1; }
  };
  struct Inherit2 : public Base {
    virtual int val() const { return 2; }
  };
}  // namespace testreftoelementid

using namespace testreftoelementid;

class testRefToElementID : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testRefToElementID);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}
  void checkAll();

  typedef std::vector<Inherit1> product1_t;
  typedef std::vector<Inherit2> product2_t;
  typedef edm::Ref<product1_t> ref1_t;
  typedef edm::Ref<product2_t> ref2_t;
  typedef edm::RefToBase<Base> reftobase_t;
};
CPPUNIT_TEST_SUITE_REGISTRATION(testRefToElementID);

void testRefToElementID::checkAll() {
  SimpleEDProductGetter getter;
  edm::ProductID id(1, 1U);

  auto prod = std::make_unique<product1_t>();
  prod->push_back(Inherit1());
  prod->push_back(Inherit1());
  getter.addProduct(id, std::move(prod));

  ref1_t ref0(id, 0, &getter);
  ref1_t ref1(id, 1, &getter);
  auto eid0 = refToElementID(ref0);
  auto eid1 = refToElementID(ref1);
  CPPUNIT_ASSERT(eid0.id() == ref0.id());
  CPPUNIT_ASSERT(eid0.index() == ref0.index());
  CPPUNIT_ASSERT(eid0 != eid1);

  reftobase_t rb0(ref0);
  reftobase_t rb1(ref1);
  auto eid3 = refToElementID(rb0);
  auto eid4 = refToElementID(rb1);
  CPPUNIT_ASSERT(eid3.id() == rb0.id());
  CPPUNIT_ASSERT(eid3.index() == rb0.key());
  CPPUNIT_ASSERT(eid3 != eid4);
}
