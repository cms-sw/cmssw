// 
#include "cppunit/extensions/HelperMacros.h"
#include <algorithm>
#include <iterator>
#include <iostream>
#include "DataFormats/Common/interface/OwnArray.h"

class testOwnArray : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testOwnArray);
  CPPUNIT_TEST(checkAll);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}
  void checkAll(); 
};

CPPUNIT_TEST_SUITE_REGISTRATION(testOwnArray);

namespace testOA {
  struct Dummy {
    Dummy(int n, bool * r) : value(n), ref(r) { }
    ~Dummy() { * ref = true; }
    int value;
    bool operator<(const Dummy & o) const {
      return value < o.value;
    }
  private:
    bool * ref;
  };

  struct DummyComp {
    bool operator()(const Dummy& d1, const Dummy& d2) const {
      return d1.value < d2.value;
    } 
  };

  class a {
  public:
    virtual ~a() {}
    virtual int f() const = 0;
  };

  class ClassB : public a {
  public:
    ClassB(int i) : ii(i) {memset(&waste, 0, sizeof(waste));}
    virtual ~ClassB() {}
    virtual int f() const { return ii;  }
    int ii;
  private:
    char waste[1024*1024];    
  };

  class ss {
  public:
    bool operator() (const a & a1, const a & a2) const { 
      return (a1.f() > a2.f());
    }
  };
  
  inline std::ostream& operator<<(std::ostream& os, const a & aa) {
    os << aa.f();
    return os;
  }
}

void testOwnArray::checkAll() {
  edm::OwnArray<testOA::Dummy,5> v;
  CPPUNIT_ASSERT(v.size() == 0);
  CPPUNIT_ASSERT(v.empty());
  bool deleted[4] = { false, false, false, false };
  v.push_back(new testOA::Dummy(0, deleted + 0));
  v.push_back(new testOA::Dummy(1, deleted + 1));
  v.push_back(new testOA::Dummy(2, deleted + 2));
  v.push_back(new testOA::Dummy(3, deleted + 3));
  CPPUNIT_ASSERT(v.size() == 4);
  auto i = v.begin();
  auto ci = i;
  * ci;
  v.sort();
  v.sort(testOA::DummyComp());
  CPPUNIT_ASSERT(!v.empty());
  CPPUNIT_ASSERT(v[0].value == 0);
  CPPUNIT_ASSERT(v[1].value == 1);
  CPPUNIT_ASSERT(v[2].value == 2);
  CPPUNIT_ASSERT(v[3].value == 3);
  i = v.begin() + 1;
  v.erase( i );
  CPPUNIT_ASSERT(!deleted[0]);
  CPPUNIT_ASSERT(deleted[1]);
  CPPUNIT_ASSERT(!deleted[2]);
  CPPUNIT_ASSERT(!deleted[3]);
  CPPUNIT_ASSERT(v.size() == 3);
  CPPUNIT_ASSERT(v[0].value == 0);
  CPPUNIT_ASSERT(v[1].value == 2);
  CPPUNIT_ASSERT(v[2].value == 3);
  auto b = v.begin(), e = b + 1;
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
  {
    edm::OwnArray<testOA::a,5> v;
    testOA::a * aa = new testOA::ClassB(2);
    v.push_back(aa);
    aa = new testOA::ClassB(1);
    v.push_back(aa);
    aa = new testOA::ClassB(3);
    v.push_back(aa);
    v.sort(testOA::ss());
    std::cout << "OwnArray : dumping contents" << std::endl;
    std::copy(v.begin(), 
	      v.end(), 
	      std::ostream_iterator<testOA::a>(std::cout, "\t"));

    edm::Ptr<testOA::a> ptr_v;
    unsigned long index(0);
    void const * data = &v[0];
    v.setPtr( typeid(testOA::a), index, data );
    testOA::a const * data_a = static_cast<testOA::a const *>(data);
    testOA::ClassB const * data_b = dynamic_cast<testOA::ClassB const *>(data_a);
    CPPUNIT_ASSERT( data != 0);
    CPPUNIT_ASSERT( data_a != 0);
    CPPUNIT_ASSERT( data_b != 0);
    if(data_b != 0) { // To silence Coverity
      CPPUNIT_ASSERT( data_b->f() == 3);
    }
  }
}
