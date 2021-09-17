/*
 *  ref_t.cppunit.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 02/11/05.
 *
 */

#include "cppunit/extensions/HelperMacros.h"
#include <vector>
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefToBaseProd.h"
#include <iostream>
using namespace edm;

class testRef : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testRef);

  CPPUNIT_TEST(constructTest);
  CPPUNIT_TEST(comparisonTest);
  CPPUNIT_TEST(getTest);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}

  void constructTest();
  void comparisonTest();
  void getTest();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testRef);

namespace {
  struct Dummy {
    Dummy() {}
    ~Dummy() {}
    bool operator==(Dummy const& iRHS) const { return this == &iRHS; }
    void const* address() const { return this; }
  };

  typedef std::vector<Dummy> DummyCollection;
}  // namespace

void testRef::constructTest() {
  Ref<DummyCollection> nulled;
  CPPUNIT_ASSERT(!nulled);
  CPPUNIT_ASSERT(nulled.isNull());
  CPPUNIT_ASSERT(!nulled.isNonnull());

  Ref<DummyCollection> nulledP;
  CPPUNIT_ASSERT(!nulledP);
  CPPUNIT_ASSERT(nulledP.isNull());
  CPPUNIT_ASSERT(!nulledP.isNonnull());

  ProductID const pid(1, 1);

  unsigned int const key = 2;
  Dummy const dummy;
  DummyCollection dummyCollection;
  dummyCollection.push_back(dummy);
  dummyCollection.push_back(dummy);
  dummyCollection.push_back(dummy);
  OrphanHandle<DummyCollection> handle(&dummyCollection, pid);
  Ref<DummyCollection> dummyRef(handle, key);
  RefProd<DummyCollection> dummyRefProd(handle);

  CPPUNIT_ASSERT(dummyRef.id() == pid);
  CPPUNIT_ASSERT(dummyRefProd.id() == pid);
  CPPUNIT_ASSERT(dummyRef.key() == key);
  CPPUNIT_ASSERT(&(*dummyRef) == &dummyCollection[key]);
  CPPUNIT_ASSERT((dummyRef.operator->()) == &dummyCollection[key]);
  CPPUNIT_ASSERT(dummyRef->address() == dummyCollection[key].address());
  CPPUNIT_ASSERT(&(*dummyRefProd) == &dummyCollection);
  CPPUNIT_ASSERT((dummyRefProd.operator->()) == &dummyCollection);

  Ref<DummyCollection> testRef1(pid, &dummyCollection[key], key, nullptr);
  CPPUNIT_ASSERT(testRef1.get() == &dummyCollection[key] && testRef1.key() == key && testRef1.isTransient() == false &&
                 testRef1.id() == pid);

  Ref<DummyCollection> testRef2(pid, &dummyCollection[key], key);
  CPPUNIT_ASSERT(testRef2.get() == &dummyCollection[key] && testRef2.key() == key && testRef2.isTransient() == false &&
                 testRef2.id() == pid);

  Ref<DummyCollection> testRef3(pid, &dummyCollection[key], key, false);
  CPPUNIT_ASSERT(testRef3.get() == &dummyCollection[key] && testRef3.key() == key && testRef3.isTransient() == false &&
                 testRef3.id() == pid);

  Ref<DummyCollection> testRef4(pid, &dummyCollection[key], key, true);
  CPPUNIT_ASSERT(testRef4.get() == &dummyCollection[key] && testRef4.key() == key && testRef4.isTransient() == true &&
                 testRef4.id() == pid);

  Ref<DummyCollection> testRef5(&dummyCollection, key);
  CPPUNIT_ASSERT(testRef5.get() == &dummyCollection[key] && testRef5.key() == key && testRef5.isTransient() == true &&
                 testRef5.id() == edm::ProductID());
}

void testRef::comparisonTest() {
  {
    ProductID const pid(1, 1);

    unsigned int const key = 2;
    Dummy const dummy;
    DummyCollection dummyCollection;
    dummyCollection.push_back(dummy);
    OrphanHandle<DummyCollection> handle(&dummyCollection, pid);
    OrphanHandle<DummyCollection> handle2(&dummyCollection, pid);
    Ref<DummyCollection> dummyRef1(handle, key);
    Ref<DummyCollection> dummyRef2(handle2, key);
    RefProd<DummyCollection> dummyRefProd1(handle);
    RefProd<DummyCollection> dummyRefProd2(handle2);

    CPPUNIT_ASSERT(dummyRef1 == dummyRef2);
    CPPUNIT_ASSERT(!(dummyRef1 != dummyRef2));
    CPPUNIT_ASSERT(!(dummyRef1 < dummyRef2));
    CPPUNIT_ASSERT(!(dummyRef2 < dummyRef1));
    CPPUNIT_ASSERT(!(dummyRef1 < dummyRef1));
    CPPUNIT_ASSERT(!(dummyRef2 < dummyRef2));

    CPPUNIT_ASSERT(dummyRefProd1 == dummyRefProd2);
    CPPUNIT_ASSERT(!(dummyRefProd1 != dummyRefProd2));
    CPPUNIT_ASSERT(!(dummyRefProd1 < dummyRefProd2));
    CPPUNIT_ASSERT(!(dummyRefProd2 < dummyRefProd1));
    CPPUNIT_ASSERT(!(dummyRefProd1 < dummyRefProd1));
    CPPUNIT_ASSERT(!(dummyRefProd2 < dummyRefProd2));

    Ref<DummyCollection> dummyRefNewKey(handle, key + 1);
    CPPUNIT_ASSERT(!(dummyRef1 == dummyRefNewKey));
    CPPUNIT_ASSERT(dummyRef1 != dummyRefNewKey);
    CPPUNIT_ASSERT(dummyRef1 < dummyRefNewKey);
    CPPUNIT_ASSERT(!(dummyRefNewKey < dummyRef1));

    ProductID const pidOther(1, 4);
    OrphanHandle<DummyCollection> handleNewPID(&dummyCollection, pidOther);
    Ref<DummyCollection> dummyRefNewPID(handleNewPID, key);
    RefProd<DummyCollection> dummyRefProdNewPID(handleNewPID);
    CPPUNIT_ASSERT(!(dummyRef1 == dummyRefNewPID));
    CPPUNIT_ASSERT(dummyRef1 != dummyRefNewPID);
    CPPUNIT_ASSERT(!(dummyRefProd1 == dummyRefProdNewPID));
    CPPUNIT_ASSERT(dummyRefProd1 != dummyRefProdNewPID);
    CPPUNIT_ASSERT(dummyRefProd1 < dummyRefProdNewPID);
    CPPUNIT_ASSERT(!(dummyRefProdNewPID < dummyRefProd1));
  }
  {
    typedef std::map<int, double> DummyCollection2;
    ProductID const pid2(1, 2);
    DummyCollection2 dummyCollection2;
    dummyCollection2.insert(std::make_pair(1, 1.0));
    dummyCollection2.insert(std::make_pair(2, 2.0));
    OrphanHandle<DummyCollection2> handle2(&dummyCollection2, pid2);
    Ref<DummyCollection2> dummyRef21(handle2, 1);
    Ref<DummyCollection2> dummyRef22(handle2, 2);
    CPPUNIT_ASSERT(dummyRef21 != dummyRef22);
    CPPUNIT_ASSERT(dummyRef21 < dummyRef22);
    CPPUNIT_ASSERT(!(dummyRef22 < dummyRef21));

    typedef std::map<int, double, std::greater<int>> DummyCollection3;
    ProductID const pid3(1, 3);
    DummyCollection3 dummyCollection3;
    dummyCollection3.insert(std::make_pair(1, 1.0));
    dummyCollection3.insert(std::make_pair(2, 2.0));
    OrphanHandle<DummyCollection3> handle3(&dummyCollection3, pid3);
    Ref<DummyCollection3> dummyRef31(handle3, 1);
    Ref<DummyCollection3> dummyRef32(handle3, 2);
    CPPUNIT_ASSERT(dummyRef31 != dummyRef32);
    CPPUNIT_ASSERT(!(dummyRef31 < dummyRef32));
    CPPUNIT_ASSERT(dummyRef32 < dummyRef31);
  }
}

namespace {
  struct TestGetter : public edm::EDProductGetter {
    WrapperBase const* hold_;
    WrapperBase const* getIt(ProductID const&) const override { return hold_; }

    std::optional<std::tuple<edm::WrapperBase const*, unsigned int>> getThinnedProduct(ProductID const&,
                                                                                       unsigned int) const override {
      return std::nullopt;
    }

    void getThinnedProducts(ProductID const& pid,
                            std::vector<WrapperBase const*>& wrappers,
                            std::vector<unsigned int>& keys) const override {}

    edm::OptionalThinnedKey getThinnedKeyFrom(ProductID const&, unsigned int, ProductID const&) const override {
      return std::monostate{};
    }

    unsigned int transitionIndex_() const override { return 0U; }
    TestGetter() : hold_(nullptr) {}
  };

  struct IntValue {
    int value_;
    IntValue(int iValue) : value_(iValue) {}
  };
}  // namespace

void testRef::getTest() {
  typedef std::vector<IntValue> IntCollection;
  auto ptr = std::make_unique<IntCollection>();

  ptr->push_back(0);
  ptr->push_back(1);

  edm::Wrapper<IntCollection> wrapper(std::move(ptr));
  TestGetter tester;
  tester.hold_ = &wrapper;

  ProductID const pid(1, 1);

  IntCollection const* wptr = reinterpret_cast<IntCollection const*>(wrapper.product());

  OrphanHandle<IntCollection> handle(wptr, pid);

  Ref<IntCollection> ref0(pid, 0, &tester);
  CPPUNIT_ASSERT(!ref0.hasProductCache());

  Ref<IntCollection> ref1(pid, 1, &tester);

  CPPUNIT_ASSERT(0 == ref0->value_);
  CPPUNIT_ASSERT(ref0.hasProductCache());
  CPPUNIT_ASSERT(1 == ref1->value_);
  CPPUNIT_ASSERT(1 == (*ref1).value_);

  Ref<IntCollection> ref0FromHandle(handle, 0);
  CPPUNIT_ASSERT(0 == ref0FromHandle->value_);
  Ref<IntCollection> ref1FromHandle(handle, 1);
  CPPUNIT_ASSERT(1 == ref1FromHandle->value_);

  RefProd<IntCollection> refProd0(handle);
  refProd0.refCore().setProductGetter(&tester);
  //refProd0.refCore().setProductPtr(0);

  RefProd<IntCollection> refProd2(pid, &tester);

  CPPUNIT_ASSERT(0 == (*refProd0)[0].value_);
  CPPUNIT_ASSERT(1 == (*refProd0)[1].value_);
  CPPUNIT_ASSERT(1 == (*refProd2)[1].value_);

  //std::cerr << ">>> RefToBaseProd from RefProd" << std::endl;
  RefToBaseProd<IntValue> refToBaseProd0(refProd0);
  //std::cerr << ">>> RefToBaseProd from Handle" << std::endl;
  RefToBaseProd<IntValue> refToBaseProd2(handle);
  //std::cerr << ">>> checking View from RefToBaseProd" << std::endl;
  const View<IntValue>& vw = *refToBaseProd0;
  //std::cerr << ">>> checking View not empty" << std::endl;
  CPPUNIT_ASSERT(!vw.empty());
  //std::cerr << ">>> checking View size" << std::endl;
  CPPUNIT_ASSERT(vw.size() == 2);
  //std::cerr << ">>> checking View element #0" << std::endl;
  CPPUNIT_ASSERT(vw[0].value_ == ref0->value_);
  //std::cerr << ">>> checking View element #1" << std::endl;
  CPPUNIT_ASSERT(vw[1].value_ == ref1->value_);
  //std::cerr << ">>> RefToBaseProd from View" << std::endl;
}
