#include "cppunit/extensions/HelperMacros.h"
#include <algorithm>
#include <cstring>
#include <iterator>
#include <iostream>
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "FWCore/Utilities/interface/propagate_const.h"

class testOwnVector : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testOwnVector);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}
  void checkAll();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testOwnVector);

namespace test {
  struct Dummy {
    Dummy(int n, bool* r) : value(n), ref(r) {}
    ~Dummy() { *ref = true; }
    int value;
    bool operator<(const test::Dummy& o) const { return value < o.value; }

  private:
    edm::propagate_const<bool*> ref;
  };

  struct DummyComp {
    bool operator()(const Dummy& d1, const Dummy& d2) const { return d1.value < d2.value; }
  };

  class a {
  public:
    virtual ~a() {}
    virtual int f() const = 0;
  };

  class ClassB : public a {
  public:
    ClassB(int i) : ii(i) { memset(&waste, 0, sizeof(waste)); }
    virtual ~ClassB() {}
    virtual int f() const { return ii; }
    int ii;

  private:
    char waste[1024 * 1024];
  };

  class ss {
  public:
    bool operator()(const a& a1, const a& a2) const { return (a1.f() > a2.f()); }
  };

  std::ostream& operator<<(std::ostream& os, const a& aa) {
    os << aa.f();
    return os;
  }
}  // namespace test

void testOwnVector::checkAll() {
  {
    edm::OwnVector<test::Dummy> v;
    CPPUNIT_ASSERT(v.size() == 0);
    CPPUNIT_ASSERT(v.empty());
    bool deleted[4] = {false, false, false, false};
    v.push_back(new test::Dummy(0, deleted + 0));
    v.push_back(new test::Dummy(1, deleted + 1));
    v.push_back(new test::Dummy(2, deleted + 2));
    v.push_back(new test::Dummy(3, deleted + 3));
    CPPUNIT_ASSERT(v.size() == 4);
    edm::OwnVector<test::Dummy>::iterator i;
    i = v.begin();
    edm::OwnVector<test::Dummy>::const_iterator ci = i;
    *ci;
    v.sort();
    v.sort(test::DummyComp());
    CPPUNIT_ASSERT(!v.empty());
    CPPUNIT_ASSERT(v[0].value == 0);
    CPPUNIT_ASSERT(v[1].value == 1);
    CPPUNIT_ASSERT(v[2].value == 2);
    CPPUNIT_ASSERT(v[3].value == 3);
    i = v.begin() + 1;
    v.erase(i);
    CPPUNIT_ASSERT(!deleted[0]);
    CPPUNIT_ASSERT(deleted[1]);
    CPPUNIT_ASSERT(!deleted[2]);
    CPPUNIT_ASSERT(!deleted[3]);
    CPPUNIT_ASSERT(v.size() == 3);
    CPPUNIT_ASSERT(v[0].value == 0);
    CPPUNIT_ASSERT(v[1].value == 2);
    CPPUNIT_ASSERT(v[2].value == 3);
    edm::OwnVector<test::Dummy>::iterator b = v.begin(), e = b + 1;
    v.erase(b, e);
    CPPUNIT_ASSERT(v.size() == 2);
    CPPUNIT_ASSERT(deleted[0]);
    CPPUNIT_ASSERT(deleted[1]);
    CPPUNIT_ASSERT(!deleted[2]);
    CPPUNIT_ASSERT(!deleted[3]);
    v.clear();
    CPPUNIT_ASSERT(v.size() == 0);
    CPPUNIT_ASSERT(v.empty());
    CPPUNIT_ASSERT(deleted[0]);
    CPPUNIT_ASSERT(deleted[1]);
    CPPUNIT_ASSERT(deleted[2]);
    CPPUNIT_ASSERT(deleted[3]);
  }
  {
    edm::OwnVector<test::a> v;
    test::a* aa = new test::ClassB(2);
    v.push_back(aa);
    aa = new test::ClassB(1);
    v.push_back(aa);
    aa = new test::ClassB(3);
    v.push_back(aa);
    v.sort(test::ss());
    std::cout << "OwnVector : dumping contents" << std::endl;
    std::copy(v.begin(), v.end(), std::ostream_iterator<test::a>(std::cout, "\t"));

    edm::Ptr<test::a> ptr_v;
    unsigned long index(0);
    void const* data = &v[0];
    v.setPtr(typeid(test::a), index, data);
    test::a const* data_a = static_cast<test::a const*>(data);
    test::ClassB const* data_b = dynamic_cast<test::ClassB const*>(data_a);
    CPPUNIT_ASSERT(data != 0);
    CPPUNIT_ASSERT(data_a != 0);
    CPPUNIT_ASSERT(data_b != 0);
    if (data_b != 0) {  // To silence Coverity
      CPPUNIT_ASSERT(data_b->f() == 3);
    }
  }
}
