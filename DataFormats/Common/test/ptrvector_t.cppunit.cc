
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/Common/interface/TestHandle.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/IntValues.h"

#include "cppunit/extensions/HelperMacros.h"

#include <algorithm>
#include <iostream>
#include <vector>

class testPtrVector : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testPtrVector);
  CPPUNIT_TEST(check);
  CPPUNIT_TEST(get);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp() {}
  void tearDown() {}
  void check();
  void get();
};

CPPUNIT_TEST_SUITE_REGISTRATION(testPtrVector);

namespace testPtr {
  struct Base {
    virtual ~Base() {}
    virtual int val() const=0;
    bool operator==(Base const& other) const { return val() == other.val(); }
  };

  struct Inherit1 : public Base {
    virtual int val() const { return 1;}
  };
  struct Inherit2 : public Base {
    virtual int val() const {return 2;}
  };

  struct TestGetter : public edm::EDProductGetter {
    edm::WrapperBase const* hold_;
    virtual edm::WrapperBase const* getIt(edm::ProductID const&) const override {
      return hold_;
    }
    virtual edm::WrapperBase const*
    getThinnedProduct(edm::ProductID const&, unsigned int&) const override {return nullptr;}

    virtual void
    getThinnedProducts(edm::ProductID const& pid,
                       std::vector<edm::WrapperBase const*>& wrappers,
                       std::vector<unsigned int>& keys) const { }


    virtual unsigned int transitionIndex_() const override {
    return 0U;
    }

    TestGetter() : hold_() {}
  };
}

using namespace testPtr;

void
do_some_tests(edm::PtrVector<Base> const& x) {
  edm::PtrVector<Base> copy(x);

  CPPUNIT_ASSERT(x.empty() == copy.empty());
  CPPUNIT_ASSERT(x.size() == copy.size());
  edm::PtrVector<Base>::const_iterator
    b = x.begin(), e = x.end(), cb = copy.begin(), ce = copy.end();
  CPPUNIT_ASSERT(e - b == ce - cb);
  CPPUNIT_ASSERT(std::distance(b, e) == std::distance(cb, ce));
}

void
testPtrVector::check() {
  using namespace edm;

  std::vector<Inherit1> v1(2,Inherit1());
  std::vector<Inherit2> v2(2,Inherit2());

  TestHandle<std::vector<Inherit1> > h1(&v1, ProductID(1, 1));
  PtrVector<Inherit1 > rv1;
  rv1.push_back(Ptr<Inherit1 >(h1, 0));
  rv1.push_back(Ptr<Inherit1 >(h1, 1));
  TestHandle<std::vector<Inherit2> > h2(&v2, ProductID(1, 2));
  PtrVector<Inherit2 > rv2;
  rv2.push_back(Ptr<Inherit2 >(h2, 0));
  rv2.push_back(Ptr<Inherit2 >(h2, 1));

  PtrVector<Base> empty;
  PtrVector<Base> copy_of_empty(empty);

  CPPUNIT_ASSERT(empty == copy_of_empty);

  PtrVector<Base> bv1(rv1);
  Ptr<Base> r1_0 = bv1[ 0 ];
  Ptr<Base> r1_1 = bv1[ 1 ];
  PtrVector<Base> bv2(rv2);
  Ptr<Base> r2_0 = bv2[ 0 ];
  Ptr<Base> r2_1 = bv2[ 1 ];

  CPPUNIT_ASSERT(bv1.empty() == false);
  CPPUNIT_ASSERT(bv1.size() == 2);
  CPPUNIT_ASSERT(bv2.size() == 2);
  CPPUNIT_ASSERT(r1_0->val() == 1);
  CPPUNIT_ASSERT(r1_1->val() == 1);
  CPPUNIT_ASSERT(r2_0->val() == 2);
  CPPUNIT_ASSERT(r2_1->val() == 2);

  PtrVector<Base>::const_iterator b = bv1.begin(), e = bv1.end();
  PtrVector<Base>::const_iterator i = b;
  CPPUNIT_ASSERT((*i)->val() == 1);
  CPPUNIT_ASSERT(i != e);
  CPPUNIT_ASSERT(i - b == 0);
  ++i;
  CPPUNIT_ASSERT((*i)->val() == 1);
  CPPUNIT_ASSERT(i != e);
  CPPUNIT_ASSERT(i - b == 1);
  ++ i;
  CPPUNIT_ASSERT(i == e);

  PtrVector<Base> assigned_from_bv1;
  do_some_tests(assigned_from_bv1);
  CPPUNIT_ASSERT(assigned_from_bv1.empty());
  assigned_from_bv1 = bv1;
  CPPUNIT_ASSERT(assigned_from_bv1.size() == bv1.size());
  CPPUNIT_ASSERT(std::equal(bv1.begin(), bv1.end(), assigned_from_bv1.begin()));
  CPPUNIT_ASSERT(assigned_from_bv1 == bv1);

  do_some_tests(assigned_from_bv1);

  /// creation of empty vector adding with push_back
  PtrVector<Base> bv3;
  bv3.push_back(r1_0);
  CPPUNIT_ASSERT(bv3.size() == 1);
  CPPUNIT_ASSERT(&(*r1_0) == &(*bv3[0]));
  bv3.push_back(r1_1);
  CPPUNIT_ASSERT(bv3.size() == 2);
  CPPUNIT_ASSERT(&(*r1_1) == &(*bv3[1]));

  /// clearing, then pushing in Ptr with other product ID
  bv3.clear();
  CPPUNIT_ASSERT(bv3.size() == 0);
  bv3.push_back(r2_0);
  CPPUNIT_ASSERT(bv3.size() == 1);
}

void
testPtrVector::get() {
  using namespace test_with_dictionaries;
  typedef std::vector<IntValue> IntCollection;
  std::unique_ptr<IntCollection> ptr(new IntCollection);

  ptr->push_back(0);
  ptr->push_back(1);
  ptr->push_back(2);
  ptr->push_back(3);

  edm::Wrapper<IntCollection> wrapper(std::move(ptr));
  TestGetter tester;
  tester.hold_ = &wrapper;

  edm::ProductID const pid(1, 1);

  IntCollection const* wptr = dynamic_cast<IntCollection const*>(wrapper.product());

  edm::PtrVector<IntValue> iVec;
  std::cerr << "pre push_backs" <<std::endl;
  iVec.push_back(edm::Ptr<IntValue>(pid,0,&tester));
  iVec.push_back(edm::Ptr<IntValue>(pid,2,&tester));
  std::cerr << "post push_backs" <<std::endl;

  iVec.setProductGetter(&tester);

  std::cerr << "pre size" <<std::endl;
  CPPUNIT_ASSERT(iVec.size() == 2);
  std::cerr << "pre iterator diff" <<std::endl;
  CPPUNIT_ASSERT(static_cast<size_t>(iVec.end()-iVec.begin()) == iVec.size());
  std::cerr << "address compare" <<std::endl;
  CPPUNIT_ASSERT(&(*(*(iVec.begin()))) == &(*(wptr->begin())));
  std::cerr << "pre value deref" <<std::endl;
  CPPUNIT_ASSERT((*(*(iVec.begin()))).value_ == 0);
  std::cerr << "pre value ->" <<std::endl;
  CPPUNIT_ASSERT((*(iVec.begin()))->value_ == 0);
  std::cerr << "post everything" <<std::endl;

  /*
  typedef std::vector<Inherit1> I1Collection;
  std::auto_ptr<I1Collection> ptr(new I1Collection);

  ptr->push_back(0);
  ptr->push_back(1);

  edm::Wrapper<I1Collection> wrapper(ptr);
  TestGetter tester;
  tester.hold_ = &wrapper;
  */
}
