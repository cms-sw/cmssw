/*
 *  ptr_t.cppunit.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 02/11/05.
 *
 */

#include <cppunit/extensions/HelperMacros.h>
#include <vector>
#include <set>
#include <list>
#include <deque>
#include "DataFormats/Common/interface/Ptr.h"

#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include <iostream>

#include "DataFormats/Common/test/IntValues.h"
#include <typeinfo>

using namespace edm;
using namespace std;
using namespace test_with_reflex;

class testPtr: public CppUnit::TestFixture {
   CPPUNIT_TEST_SUITE(testPtr);

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
CPPUNIT_TEST_SUITE_REGISTRATION(testPtr);

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
  };

  typedef std::vector<Dummy2> DummyCollection2;
}

void testPtr::constructTest() {
   Ptr<Dummy> nulled;
   CPPUNIT_ASSERT(!nulled);
   CPPUNIT_ASSERT(nulled.isNull());
   CPPUNIT_ASSERT(!nulled.isNonnull());
   CPPUNIT_ASSERT(!nulled.isAvailable());

   Ptr<Dummy> nulledP;
   CPPUNIT_ASSERT(!nulledP);
   CPPUNIT_ASSERT(nulledP.isNull());
   CPPUNIT_ASSERT(!nulledP.isNonnull());
   CPPUNIT_ASSERT(!nulled.isAvailable());

   ProductID const pid(1);

   {
     unsigned int const key = 2;
     Dummy const dummy;
     DummyCollection dummyContainer;
     dummyContainer.push_back(dummy);
     dummyContainer.push_back(dummy);
     dummyContainer.push_back(dummy);
     OrphanHandle<DummyCollection> handle(&dummyContainer, pid);
     Ptr<Dummy> dummyPtr(handle,key);
     
     CPPUNIT_ASSERT(dummyPtr.isAvailable());
     CPPUNIT_ASSERT(dummyPtr.id() == pid);
     //CPPUNIT_ASSERT(dummyPtrProd.id() == pid);
     CPPUNIT_ASSERT(dummyPtr.key() == key);
     CPPUNIT_ASSERT(dummyPtr.get() == &dummyContainer[key]);
     CPPUNIT_ASSERT(&(*dummyPtr) == &dummyContainer[key]);
     CPPUNIT_ASSERT((dummyPtr.operator->()) == &dummyContainer[key]);
     CPPUNIT_ASSERT(dummyPtr->address() == dummyContainer[key].address());
   }

   {
     unsigned int const key = 2;
     Dummy const dummy;
     DummySet dummyContainer;
     dummyContainer.insert(dummy);
     dummyContainer.insert(dummy);
     dummyContainer.insert(dummy);
     OrphanHandle<DummySet> handle(&dummyContainer, pid);
     Ptr<Dummy> dummyPtr(handle,key);
     
     CPPUNIT_ASSERT(dummyPtr.isAvailable());
     CPPUNIT_ASSERT(dummyPtr.id() == pid);
     //CPPUNIT_ASSERT(dummyPtrProd.id() == pid);
     CPPUNIT_ASSERT(dummyPtr.key() == key);
     DummySet::const_iterator it = dummyContainer.begin();
     std::advance(it, key);
     CPPUNIT_ASSERT(dummyPtr.get() == it->address());
     CPPUNIT_ASSERT(&(*dummyPtr) == it->address());
     CPPUNIT_ASSERT((dummyPtr.operator->()) == it->address());
     CPPUNIT_ASSERT(dummyPtr->address() == it->address());
   }

   {
     unsigned int const key = 2;
     Dummy const dummy;
     DummyList dummyContainer;
     dummyContainer.push_back(dummy);
     dummyContainer.push_back(dummy);
     dummyContainer.push_back(dummy);
     OrphanHandle<DummyList> handle(&dummyContainer, pid);
     Ptr<Dummy> dummyPtr(handle,key);
     
     CPPUNIT_ASSERT(dummyPtr.isAvailable());
     CPPUNIT_ASSERT(dummyPtr.id() == pid);
     //CPPUNIT_ASSERT(dummyPtrProd.id() == pid);
     CPPUNIT_ASSERT(dummyPtr.key() == key);
     DummyList::const_iterator it = dummyContainer.begin();
     std::advance(it, key);
     CPPUNIT_ASSERT(dummyPtr.get() == it->address());
     CPPUNIT_ASSERT(&(*dummyPtr) == it->address());
     CPPUNIT_ASSERT((dummyPtr.operator->()) == it->address());
     CPPUNIT_ASSERT(dummyPtr->address() == it->address());
   }

   {
     unsigned int const key = 2;
     Dummy const dummy;
     DummyDeque dummyContainer;
     dummyContainer.push_back(dummy);
     dummyContainer.push_back(dummy);
     dummyContainer.push_back(dummy);
     OrphanHandle<DummyDeque> handle(&dummyContainer, pid);
     Ptr<Dummy> dummyPtr(handle,key);
     
     CPPUNIT_ASSERT(dummyPtr.isAvailable());
     CPPUNIT_ASSERT(dummyPtr.id() == pid);
     //CPPUNIT_ASSERT(dummyPtrProd.id() == pid);
     CPPUNIT_ASSERT(dummyPtr.key() == key);
     DummyDeque::const_iterator it = dummyContainer.begin();
     std::advance(it, key);
     CPPUNIT_ASSERT(dummyPtr.get() == it->address());
     CPPUNIT_ASSERT(&(*dummyPtr) == it->address());
     CPPUNIT_ASSERT((dummyPtr.operator->()) == it->address());
     CPPUNIT_ASSERT(dummyPtr->address() == it->address());
   }
   
   {
     unsigned int const key = 2;
     Dummy2 const dummy;
     DummyCollection2 dummyContainer;
     dummyContainer.push_back(dummy);
     dummyContainer.push_back(dummy);
     dummyContainer.push_back(dummy);
     OrphanHandle<DummyCollection2> handle(&dummyContainer, pid);
     Ptr<Dummy> dummyPtr(handle,key);
     Ptr<Dummy2> dummyPtr2(dummyPtr);
     
     CPPUNIT_ASSERT(dummyPtr.isAvailable());
     CPPUNIT_ASSERT(dummyPtr.id() == pid);
     //CPPUNIT_ASSERT(dummyPtrProd.id() == pid);
     CPPUNIT_ASSERT(dummyPtr.key() == key);
     CPPUNIT_ASSERT(dummyPtr.get() == &dummyContainer[key]);
     CPPUNIT_ASSERT(&(*dummyPtr) == &dummyContainer[key]);
     CPPUNIT_ASSERT((dummyPtr.operator->()) == &dummyContainer[key]);
     CPPUNIT_ASSERT(dummyPtr->address() == dummyContainer[key].address());

     CPPUNIT_ASSERT(dummyPtr2.isAvailable());
     CPPUNIT_ASSERT(dummyPtr2.id() == pid);
     CPPUNIT_ASSERT(dummyPtr2.key() == key);
     CPPUNIT_ASSERT(dummyPtr2.get() == &dummyContainer[key]);
     CPPUNIT_ASSERT(&(*dummyPtr2) == &dummyContainer[key]);
     CPPUNIT_ASSERT((dummyPtr2.operator->()) == &dummyContainer[key]);
     CPPUNIT_ASSERT(dummyPtr2->address() == dummyContainer[key].address());
     
     Ptr<Dummy2> dummy2Ptr(handle,key);
     Ptr<Dummy> copyPtr(dummy2Ptr);
     CPPUNIT_ASSERT(dummy2Ptr.key() == copyPtr.key());
     CPPUNIT_ASSERT(dummy2Ptr.get() == static_cast<const Dummy*>(dummy2Ptr.get()));
   }
   
}

void testPtr::comparisonTest() {

 {
   ProductID const pid(1);

   unsigned int const key = 2;
   Dummy const dummy;
   DummyCollection dummyContainer;
   dummyContainer.push_back(dummy);
   OrphanHandle<DummyCollection> handle(&dummyContainer, pid);
   OrphanHandle<DummyCollection> handle2(&dummyContainer, pid);
   Ptr<Dummy> dummyPtr1(handle, key);
   Ptr<Dummy> dummyPtr2(handle2, key);
   //PtrProd<DummyCollection> dummyPtrProd1(handle);
   //PtrProd<DummyCollection> dummyPtrProd2(handle2);

   CPPUNIT_ASSERT(dummyPtr1 == dummyPtr2);
   CPPUNIT_ASSERT(!(dummyPtr1 != dummyPtr2));
   CPPUNIT_ASSERT(!(dummyPtr1 < dummyPtr2));
   CPPUNIT_ASSERT(!(dummyPtr2 < dummyPtr1));
   CPPUNIT_ASSERT(!(dummyPtr1 < dummyPtr1));
   CPPUNIT_ASSERT(!(dummyPtr2 < dummyPtr2));

/*   CPPUNIT_ASSERT(dummyPtrProd1 == dummyPtrProd2);
   CPPUNIT_ASSERT(!(dummyPtrProd1 != dummyPtrProd2));
   CPPUNIT_ASSERT(!(dummyPtrProd1 < dummyPtrProd2));
   CPPUNIT_ASSERT(!(dummyPtrProd2 < dummyPtrProd1));
   CPPUNIT_ASSERT(!(dummyPtrProd1 < dummyPtrProd1));
   CPPUNIT_ASSERT(!(dummyPtrProd2 < dummyPtrProd2)); */

   Ptr<Dummy> dummyPtrNewKey(handle, key+1);
   CPPUNIT_ASSERT(!(dummyPtr1 == dummyPtrNewKey));
   CPPUNIT_ASSERT(dummyPtr1 != dummyPtrNewKey);
   CPPUNIT_ASSERT(dummyPtr1 < dummyPtrNewKey);
   CPPUNIT_ASSERT(!(dummyPtrNewKey < dummyPtr1));

   ProductID const pidOther(4);
   OrphanHandle<DummyCollection> handleNewPID(&dummyContainer, pidOther);
   Ptr<Dummy> dummyPtrNewPID(handleNewPID, key);
   //PtrProd<DummyCollection> dummyPtrProdNewPID(handleNewPID);
   CPPUNIT_ASSERT(!(dummyPtr1 == dummyPtrNewPID));
   CPPUNIT_ASSERT(dummyPtr1 != dummyPtrNewPID);
   /*CPPUNIT_ASSERT(!(dummyPtrProd1 == dummyPtrProdNewPID));
   CPPUNIT_ASSERT(dummyPtrProd1 != dummyPtrProdNewPID);
   CPPUNIT_ASSERT(dummyPtrProd1 < dummyPtrProdNewPID);
   CPPUNIT_ASSERT(!(dummyPtrProdNewPID < dummyPtrProd1)); */
 }
  /*
   
{
   typedef std::map<int, double> DummyCollection2;
   ProductID const pid2(2);
   DummyCollection2 dummyContainer2;
   dummyContainer2.insert(std::make_pair(1, 1.0));
   dummyContainer2.insert(std::make_pair(2, 2.0));
   OrphanHandle<DummyCollection2> handle2(&dummyContainer2, pid2);
   Ptr<DummyCollection2> dummyPtr21(handle2, 1);
   Ptr<DummyCollection2> dummyPtr22(handle2, 2);
   CPPUNIT_ASSERT(dummyPtr21 != dummyPtr22);
   CPPUNIT_ASSERT(dummyPtr21 < dummyPtr22);
   CPPUNIT_ASSERT(!(dummyPtr22 < dummyPtr21));

   typedef std::map<int, double, std::greater<int> > DummyCollection3;
   ProductID const pid3(3);
   DummyCollection3 dummyContainer3;
   dummyContainer3.insert(std::make_pair(1, 1.0));
   dummyContainer3.insert(std::make_pair(2, 2.0));
   OrphanHandle<DummyCollection3> handle3(&dummyContainer3, pid3);
   Ptr<DummyCollection3> dummyPtr31(handle3, 1);
   Ptr<DummyCollection3> dummyPtr32(handle3, 2);
   CPPUNIT_ASSERT(dummyPtr31 != dummyPtr32);
   CPPUNIT_ASSERT(!(dummyPtr31 < dummyPtr32));
   CPPUNIT_ASSERT(dummyPtr32 < dummyPtr31);
 }
*/
}


namespace {
  using namespace Reflex;
   struct TestGetter : public edm::EDProductGetter {
      EDProduct const* hold_;
      EDProduct const* getIt(ProductID const&) const {
         return hold_;
      }

      TestGetter() : hold_(0) {}
   };
}


void testPtr::getTest() {
   typedef std::vector<IntValue> IntCollection;
   std::auto_ptr<IntCollection> ptr(new IntCollection);

   ptr->push_back(0);
   ptr->push_back(1);

   edm::Wrapper<IntCollection> wrapper(ptr);
   TestGetter tester;
   tester.hold_ = &wrapper;

   ProductID const pid(1);

   IntCollection const* wptr = dynamic_cast<IntCollection const*>(wrapper.product());

   OrphanHandle<IntCollection> handle(wptr, pid);

   Ptr<IntValue> ref0(pid, 0,&tester);
   CPPUNIT_ASSERT( !ref0.hasCache());

   Ptr<IntValue> ref1(pid, 1, &tester);

   Ptr<IntValue> ref2(pid, 1, &tester);

   CPPUNIT_ASSERT(ref0.isAvailable());
   CPPUNIT_ASSERT(0 == ref0->value_);
   CPPUNIT_ASSERT(ref0.hasCache());
   CPPUNIT_ASSERT(1 == ref1->value_);
   CPPUNIT_ASSERT(1 == ref2->value_);
   CPPUNIT_ASSERT(1 == (*ref1).value_);

   {
     typedef std::vector<IntValue2> SDCollection;
     std::auto_ptr<SDCollection> ptr(new SDCollection);
     
     ptr->push_back(IntValue2(0));
     ptr->back().value_ = 0;
     ptr->push_back(IntValue2(1));
     ptr->back().value_ = 1;
     
     edm::Wrapper<SDCollection> wrapper(ptr);
     TestGetter tester;
     tester.hold_ = &wrapper;
     
     ProductID const pid(1);
     
     SDCollection const* wptr = dynamic_cast<SDCollection const*>(wrapper.product());
     
     OrphanHandle<SDCollection> handle(wptr, pid);
     
     Ptr<IntValue> ref0(pid, 0,&tester);
     CPPUNIT_ASSERT( !ref0.hasCache());
     
     Ptr<IntValue> ref1(pid, 1, &tester);
     
     Ptr<IntValue> ref2(pid, 1, &tester);
     
     CPPUNIT_ASSERT(0 == ref0->value_);
     CPPUNIT_ASSERT(ref0.hasCache());
     CPPUNIT_ASSERT(1 == ref1->value_);
     CPPUNIT_ASSERT(1 == ref2->value_);
     CPPUNIT_ASSERT(1 == (*ref1).value_);
   }
   
   {
      TestGetter tester;
      tester.hold_ = 0;
      ProductID const pid(1);

      Ptr<IntValue> ref0(pid, 0,&tester);
      CPPUNIT_ASSERT_THROW((*ref0),cms::Exception);
      CPPUNIT_ASSERT_THROW((ref0.operator->()),cms::Exception);
   }
   /*
   PtrProd<IntCollection> refProd0(handle);
   refProd0.refCore().setProductGetter(&tester);
   refProd0.refCore().setProductPtr(0);

   PtrProd<IntCollection> refProd2(pid, &tester);

   CPPUNIT_ASSERT(0 == (*refProd0)[0].value_);
   CPPUNIT_ASSERT(1 == (*refProd0)[1].value_);
   CPPUNIT_ASSERT(1 == (*refProd2)[1].value_);

   //get it via the 'singleton'
   edm::EDProductGetter::Operate operate(&tester);
   Ptr<IntCollection> ref0b(handle, 0);
   ref0b.ref().refCore().setProductPtr(0);
   ref0b.ref().item().setPtr(0);
   CPPUNIT_ASSERT(0 == ref0b->value_);

   PtrProd<IntCollection> refProd0b(handle);
   refProd0b.refCore().setProductPtr(0);
   CPPUNIT_ASSERT(1 == (*refProd0b)[1].value_);

   cerr << ">>> PtrToBaseProd from PtrProd" << endl;
   PtrToBaseProd<IntValue> refToBaseProd0(refProd0);
   cerr << ">>> PtrToBaseProd from Ptr" << endl;
   PtrToBaseProd<IntValue> refToBaseProd1(ref0);
   cerr << ">>> PtrToBaseProd from Handle" << endl;
   PtrToBaseProd<IntValue> refToBaseProd2(handle);
   cerr << ">>> checking View from PtrToBaseProd" << endl;
   const View<IntValue> & vw = * refToBaseProd0;
   cerr << ">>> checking View not empty" << endl;
   CPPUNIT_ASSERT( ! vw.empty() );
   cerr << ">>> checking View size" << endl;
   CPPUNIT_ASSERT( vw.size() == 2 );
   cerr << ">>> checking View element #0" << endl;
   CPPUNIT_ASSERT( vw[0].value_ == ref0->value_ );
   cerr << ">>> checking View element #1" << endl;
   CPPUNIT_ASSERT( vw[1].value_ == ref1->value_ );
   cerr << ">>> PtrToBaseProd from View" << endl;
   PtrToBaseProd<IntValue> refToBaseProd3(vw);
   cerr << ">>> checking ref. not empty" << endl;
   CPPUNIT_ASSERT( ! refToBaseProd3->empty() );
   cerr << ">>> checking ref. size" << endl;
   CPPUNIT_ASSERT( refToBaseProd3->size() == 2 );
   cerr << ">>> checking ref. element #0" << endl;
   CPPUNIT_ASSERT( (*refToBaseProd3)[0].value_ == ref0->value_ );
   cerr << ">>> checking ref. element #1" << endl;
   CPPUNIT_ASSERT( (*refToBaseProd3)[1].value_ == ref1->value_ );
    */
}
