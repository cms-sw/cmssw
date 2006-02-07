/*
 *  ref_t.cppunit.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 02/11/05.
 *
 */

#include <cppunit/extensions/HelperMacros.h>
#include <vector>
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/Common/test/TestHandle.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"

using namespace edm;

class testRef: public CppUnit::TestFixture {
   CPPUNIT_TEST_SUITE(testRef);
   
   CPPUNIT_TEST(constructTest);
   CPPUNIT_TEST(comparisonTest);
   CPPUNIT_TEST(getTest);
   
   CPPUNIT_TEST_SUITE_END();
public:
      void setUp(){}
   void tearDown(){}
   
   void constructTest();
   void comparisonTest();
   void getTest();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testRef);

namespace {
   struct Dummy { 
      Dummy() {} bool 
      operator==(Dummy const& iRHS) const {return this == &iRHS;} 
      void const* address() const {return this;}
   };
   typedef std::vector<Dummy> DummyCollection;
}

void testRef::constructTest() {
   Ref<DummyCollection> nulled;
   CPPUNIT_ASSERT(!nulled);
   CPPUNIT_ASSERT(!static_cast<bool>(nulled));
   CPPUNIT_ASSERT(nulled.isNull());
   CPPUNIT_ASSERT(!nulled.isNonnull());
   
   Ref<DummyCollection> nulledP;
   CPPUNIT_ASSERT(!nulledP);
   CPPUNIT_ASSERT(!static_cast<bool>(nulledP));
   CPPUNIT_ASSERT(nulledP.isNull());
   CPPUNIT_ASSERT(!nulledP.isNonnull());
   
   ProductID const pid(1);
   
   unsigned int const index = 2;
   Dummy const dummy;
   DummyCollection dummyCollection;
   dummyCollection.push_back(dummy);
   dummyCollection.push_back(dummy);
   dummyCollection.push_back(dummy);
   TestHandle<DummyCollection> handle(&dummyCollection, pid);
   Ref<DummyCollection> dummyRef(handle,index);
   RefProd<DummyCollection> dummyRefProd(handle);
   
   CPPUNIT_ASSERT(dummyRef.id() == pid);
   CPPUNIT_ASSERT(dummyRefProd.id() == pid);
   CPPUNIT_ASSERT(dummyRef.ref().item().index() == index);
   CPPUNIT_ASSERT(dummyRef.ref().item().ptr() == &dummyCollection[index]);
   CPPUNIT_ASSERT(dummyRef.index() == index);
   CPPUNIT_ASSERT(dummyRef.product() == &dummyCollection);
   CPPUNIT_ASSERT(&(*dummyRef) == &dummyCollection[index]);
   CPPUNIT_ASSERT((dummyRef.operator->()) == &dummyCollection[index]);
   CPPUNIT_ASSERT(dummyRef->address() == dummyCollection[index].address());
   CPPUNIT_ASSERT(&(*dummyRefProd) == &dummyCollection);
   CPPUNIT_ASSERT((dummyRefProd.operator->()) == &dummyCollection);
}

void testRef::comparisonTest() {
   
   ProductID const pid(1);
   
   unsigned int const index = 2;
   Dummy const dummy;
   DummyCollection dummyCollection;
   dummyCollection.push_back(dummy);
   TestHandle<DummyCollection> handle(&dummyCollection, pid);
   TestHandle<DummyCollection> handle2(&dummyCollection, pid);
   Ref<DummyCollection> dummyRef1(handle,index);
   Ref<DummyCollection> dummyRef2(handle2,index);
   RefProd<DummyCollection> dummyRefProd1(handle);
   RefProd<DummyCollection> dummyRefProd2(handle2);

   CPPUNIT_ASSERT(dummyRef1 == dummyRef2);
   CPPUNIT_ASSERT(!(dummyRef1 != dummyRef2));

   CPPUNIT_ASSERT(dummyRefProd1 == dummyRefProd2);
   CPPUNIT_ASSERT(!(dummyRefProd1 != dummyRefProd2));

   Ref<DummyCollection> dummyRefNewIndex(handle, index+1);
   CPPUNIT_ASSERT(dummyRef1 != dummyRefNewIndex);
   
   ProductID const pidOther(2);
   TestHandle<DummyCollection> handleNewPID(&dummyCollection, pidOther);
   Ref<DummyCollection> dummyRefNewPID(handleNewPID, index);
   RefProd<DummyCollection> dummyRefProdNewPID(handleNewPID);
   CPPUNIT_ASSERT(dummyRef1 != dummyRefNewPID);
   CPPUNIT_ASSERT(dummyRefProd1 != dummyRefProdNewPID);
}

namespace {
   struct TestGetter : public edm::EDProductGetter {
      EDProduct const* hold_;
      EDProduct const* getIt(ProductID const&) const {
         return hold_;
      }

      TestGetter() : hold_(0) {}
   };
   
   struct IntValue {
      int value_;
      IntValue(int iValue): value_(iValue) {}
   };
}

void testRef::getTest() {
   typedef std::vector<IntValue> IntCollection;
   std::auto_ptr<IntCollection> ptr(new IntCollection);

   ptr->push_back(0);
   ptr->push_back(1);
   
   edm::Wrapper<IntCollection> wrapper(ptr);
   TestGetter tester;
   tester.hold_ = &wrapper;

   ProductID const pid(1);

   IntCollection const* wptr = dynamic_cast<IntCollection const*>(wrapper.product());

   TestHandle<IntCollection> handle(wptr, pid);

   Ref<IntCollection> ref0(handle, 0);
   ref0.ref().product().setProductGetter(&tester);
   ref0.ref().product().setProductPtr(0);
   ref0.ref().item().setPtr(0);
   
   Ref<IntCollection> ref1(handle, 1);
   ref1.ref().product().setProductGetter(&tester);
   ref1.ref().product().setProductPtr(0);
   ref1.ref().item().setPtr(0);

   Ref<IntCollection> ref2(pid, 1, &tester);

   CPPUNIT_ASSERT(0 == ref0->value_);
   CPPUNIT_ASSERT(1 == ref1->value_);
   CPPUNIT_ASSERT(1 == ref2->value_);
   CPPUNIT_ASSERT(1 == (*ref1).value_);

   RefProd<IntCollection> refProd0(handle);
   refProd0.product().setProductGetter(&tester);
   refProd0.product().setProductPtr(0);

   RefProd<IntCollection> refProd2(pid, &tester);

   CPPUNIT_ASSERT(0 == (*refProd0)[0].value_);
   CPPUNIT_ASSERT(1 == (*refProd0)[1].value_);
   CPPUNIT_ASSERT(1 == (*refProd2)[1].value_);

   //get it via the 'singleton'
   edm::EDProductGetter::Operate operate(&tester);
   Ref<IntCollection> ref0b(handle, 0);
   ref0b.ref().product().setProductPtr(0);
   ref0b.ref().item().setPtr(0);
   CPPUNIT_ASSERT(0 == ref0b->value_);

   RefProd<IntCollection> refProd0b(handle);
   refProd0b.product().setProductPtr(0);
   CPPUNIT_ASSERT(1 == (*refProd0b)[1].value_);
}
