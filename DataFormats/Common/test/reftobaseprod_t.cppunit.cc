/*
 *  reftobaseprod_t.cppunit.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 02/11/05.
 *
 */

#include "cppunit/extensions/HelperMacros.h"
#include <vector>
#include <set>
#include <list>
#include <deque>
#include "DataFormats/Common/interface/RefToBaseProd.h"

#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include <iostream>

#include "DataFormats/Common/interface/IntValues.h"
#include <typeinfo>

using namespace edm;
using namespace test_with_dictionaries;

class testRefToBaseProd: public CppUnit::TestFixture {
   CPPUNIT_TEST_SUITE(testRefToBaseProd);

   CPPUNIT_TEST(constructTest);
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
CPPUNIT_TEST_SUITE_REGISTRATION(testRefToBaseProd);

namespace {
  struct Dummy {
    Dummy() {}
    virtual ~Dummy() {}
    bool operator==(Dummy const& iRHS) const {return this == &iRHS;}
    bool operator<(Dummy const& iRHS) const { return this->address() < iRHS.address(); }
    void const* address() const {return this;}
  };

  typedef std::vector<Dummy> DummyCollection;

  typedef std::set<Dummy> DummySet;
  typedef std::list<Dummy> DummyList;
  typedef std::deque<Dummy> DummyDeque;
  
  struct Dummy2 : public Dummy {
    Dummy2() {}
    virtual ~Dummy2() {}

  };

  typedef std::vector<Dummy2> DummyCollection2;
  
  template< typename T, typename C> void compareTo( const RefToBaseProd<T>& iProd, C const& iContainer) {

    unsigned int index=0;
    for( auto const& item: iContainer) {
      CPPUNIT_ASSERT(&((*iProd.get())[index++]) == &item);
    }
    index=0;
    for( auto const& item: iContainer) {
      CPPUNIT_ASSERT(&((*iProd)[index++]) == &item);
    }
    index=0;
    for( auto const& item: iContainer) {
      CPPUNIT_ASSERT(&((*(iProd.operator->()))[index++]) == &item);
    }
    index=0;
    for( auto const& item: iContainer) {
      CPPUNIT_ASSERT(&(iProd->at(index++)) == &item);
    }
  }
}

void testRefToBaseProd::constructTest() {
   RefToBaseProd<Dummy> nulled;
   CPPUNIT_ASSERT(!nulled);
   CPPUNIT_ASSERT(nulled.isNull());
   CPPUNIT_ASSERT(!nulled.isNonnull());

   RefToBaseProd<Dummy> nulledP;
   CPPUNIT_ASSERT(!nulledP);
   CPPUNIT_ASSERT(nulledP.isNull());
   CPPUNIT_ASSERT(!nulledP.isNonnull());

   ProductID const pid(1, 1);

   {
     Dummy const dummy;
     DummyCollection dummyContainer;
     dummyContainer.push_back(dummy);
     dummyContainer.push_back(dummy);
     dummyContainer.push_back(dummy);
     OrphanHandle<DummyCollection> handle(&dummyContainer, pid);
     RefToBaseProd<Dummy> dummyPtr(handle);
     
     CPPUNIT_ASSERT(dummyPtr.id() == pid);
     compareTo(dummyPtr,dummyContainer);
   }

   {
     Dummy const dummy;
     DummySet dummyContainer;
     dummyContainer.insert(dummy);
     dummyContainer.insert(dummy);
     dummyContainer.insert(dummy);
     OrphanHandle<DummySet> handle(&dummyContainer, pid);
     RefToBaseProd<Dummy> dummyPtr(handle);
     
     CPPUNIT_ASSERT(dummyPtr.id() == pid);
     compareTo(dummyPtr,dummyContainer);
   }

   {
     Dummy const dummy;
     DummyList dummyContainer;
     dummyContainer.push_back(dummy);
     dummyContainer.push_back(dummy);
     dummyContainer.push_back(dummy);
     OrphanHandle<DummyList> handle(&dummyContainer, pid);
     RefToBaseProd<Dummy> dummyPtr(handle);
     
     CPPUNIT_ASSERT(dummyPtr.id() == pid);
     compareTo(dummyPtr,dummyContainer);
   }

   {
     Dummy const dummy;
     DummyDeque dummyContainer;
     dummyContainer.push_back(dummy);
     dummyContainer.push_back(dummy);
     dummyContainer.push_back(dummy);
     OrphanHandle<DummyDeque> handle(&dummyContainer, pid);
     RefToBaseProd<Dummy> dummyPtr(handle);
     
     CPPUNIT_ASSERT(dummyPtr.id() == pid);
     compareTo(dummyPtr,dummyContainer);
   }
   
   {
     Dummy2 const dummy;
     DummyCollection2 dummyContainer;
     dummyContainer.push_back(dummy);
     dummyContainer.push_back(dummy);
     dummyContainer.push_back(dummy);
     OrphanHandle<DummyCollection2> handle(&dummyContainer, pid);
     RefToBaseProd<Dummy> dummyPtr(handle);
     RefToBaseProd<Dummy> dummyPtr2(dummyPtr);
     
     CPPUNIT_ASSERT(dummyPtr.id() == pid);
     compareTo(dummyPtr,dummyContainer);

     CPPUNIT_ASSERT(dummyPtr2.id() == pid);
     compareTo(dummyPtr2,dummyContainer);     
   }
   
}

namespace {
   struct TestGetter : public edm::EDProductGetter {
      WrapperBase const* hold_;
      virtual WrapperBase const* getIt(ProductID const&) const override {
         return hold_;
      }
      virtual unsigned int transitionIndex_() const override {
         return 0U;
      }

      TestGetter() : hold_() {}
   };
}

void testRefToBaseProd::getTest() {
   typedef std::vector<IntValue> IntCollection;
   std::unique_ptr<IntCollection> ptr(new IntCollection);

   ptr->push_back(0);
   ptr->push_back(1);

   edm::Wrapper<IntCollection> wrapper(std::move(ptr));
   TestGetter tester;
   tester.hold_ = &wrapper;

   ProductID const pid(1, 1);

   IntCollection const* wptr = dynamic_cast<IntCollection const*>(wrapper.product());

   OrphanHandle<IntCollection> handle(wptr, pid);

   //NOTE: ROOT will touch the private variables directly and since the RefToBaseProd
   // has no constructor which takes RefCore, I have to play this dirty trick
   assert(sizeof(edm::RefCore)==sizeof(edm::RefToBaseProd<IntValue>));
   
   RefCore core(pid,nullptr,&tester,false);
   RefToBaseProd<IntValue>& prod = reinterpret_cast<RefToBaseProd<IntValue>&>(core);

   //previously making a copy before reading back would cause seg fault
   RefToBaseProd<IntValue> prodCopy(prod);
   
   CPPUNIT_ASSERT(!prod.hasCache());

   CPPUNIT_ASSERT(0 != prod.get());
   CPPUNIT_ASSERT(prod.hasCache());
   compareTo(prod,*wptr);

   
   CPPUNIT_ASSERT(!prodCopy.hasCache());
   
   CPPUNIT_ASSERT(0 != prodCopy.get());
   CPPUNIT_ASSERT(prodCopy.hasCache());
   compareTo(prodCopy,*wptr);

   {
      typedef std::vector<IntValue2> SDCollection;
      std::unique_ptr<SDCollection> ptr(new SDCollection);

      ptr->push_back(IntValue2(0));
      ptr->back().value_ = 0;
      ptr->push_back(IntValue2(1));
      ptr->back().value_ = 1;

      edm::Wrapper<SDCollection> wrapper(std::move(ptr));
      TestGetter tester;
      tester.hold_ = &wrapper;

      ProductID const pid(1, 1);

      SDCollection const* wptr = dynamic_cast<SDCollection const*>(wrapper.product());

      OrphanHandle<SDCollection> handle(wptr, pid);

      //NOTE: ROOT will touch the private variables directly and since the RefToBaseProd
      // has no constructor which takes RefCore, I have to play this dirty trick
      assert(sizeof(edm::RefCore)==sizeof(edm::RefToBaseProd<IntValue>));

      RefCore core(pid,nullptr,&tester,false);
      RefToBaseProd<IntValue>& prod = reinterpret_cast<RefToBaseProd<IntValue>&>(core);
      
      CPPUNIT_ASSERT(!prod.hasCache());
      
      CPPUNIT_ASSERT(0 != prod.get());
      CPPUNIT_ASSERT(prod.hasCache());
      compareTo(prod,*wptr);

   }
   
   {
      TestGetter tester;
      tester.hold_ = nullptr;
      ProductID const pid(1, 1);

      //NOTE: ROOT will touch the private variables directly and since the RefToBaseProd
      // has no constructor which takes RefCore, I have to play this dirty trick
      assert(sizeof(edm::RefCore)==sizeof(edm::RefToBaseProd<IntValue>));
      
      RefCore core(pid,nullptr,&tester,false);
      RefToBaseProd<IntValue>& prod = reinterpret_cast<RefToBaseProd<IntValue>&>(core);

      CPPUNIT_ASSERT_THROW((*prod),cms::Exception);
      CPPUNIT_ASSERT_THROW((prod.operator->()),cms::Exception);
   }
}
