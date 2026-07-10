/*
 *  ptr_t.cppunit.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 02/11/05.
 *
 */

#include <catch2/catch_all.hpp>
#include <vector>
#include <set>
#include <list>
#include <deque>
#include "DataFormats/Common/interface/Ptr.h"

#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/Common/interface/OrphanHandle.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include <iostream>

#include "DataFormats/Common/interface/IntValues.h"
#include <typeinfo>

using namespace edm;
using namespace test_with_dictionaries;

namespace {
  struct Dummy {
    Dummy() {}
    virtual ~Dummy() {}
    bool operator==(Dummy const& iRHS) const { return this == &iRHS; }
    bool operator<(Dummy const& iRHS) const { return this->address() < iRHS.address(); }
    void const* address() const { return this; }
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

  struct TestGetter : public edm::EDProductGetter {
    WrapperBase const* hold_;
    WrapperBase const* getIt(ProductID const&) const override { return hold_; }
    unsigned int transitionIndex_() const override { return 0U; }

    TestGetter() : hold_() {}
  };
}  // namespace

TEST_CASE("Ptr", "[Ptr]") {
  SECTION("construction") {
    Ptr<Dummy> nulled;
    REQUIRE(!nulled);
    REQUIRE(nulled.isNull());
    REQUIRE(!nulled.isNonnull());
    REQUIRE(!nulled.isAvailable());

    ProductID const pid(1, 1);

    SECTION("vector container") {
      unsigned int const key = 2;
      Dummy const dummy;
      DummyCollection dummyContainer;
      dummyContainer.push_back(dummy);
      dummyContainer.push_back(dummy);
      dummyContainer.push_back(dummy);
      OrphanHandle<DummyCollection> handle(&dummyContainer, pid);
      Ptr<Dummy> dummyPtr(handle, key);

      REQUIRE(dummyPtr.isAvailable());
      REQUIRE(dummyPtr.id() == pid);
      REQUIRE(dummyPtr.key() == key);
      REQUIRE(dummyPtr.get() == &dummyContainer[key]);
      REQUIRE(&(*dummyPtr) == &dummyContainer[key]);
      REQUIRE((dummyPtr.operator->()) == &dummyContainer[key]);
      REQUIRE(dummyPtr->address() == dummyContainer[key].address());

      Ptr<Dummy> anotherPtr(dummyPtr.id(), dummyPtr.get(), dummyPtr.key(), dummyPtr.isTransient());
      REQUIRE(dummyPtr.get() == anotherPtr.get());
      REQUIRE(dummyPtr.key() == anotherPtr.key());
      REQUIRE(dummyPtr.isTransient() == anotherPtr.isTransient());
      REQUIRE(dummyPtr.id() == anotherPtr.id());

      Ptr<Dummy> ptrTrans(&dummyContainer, key);
      Ptr<Dummy> ptrTrans2(&dummyContainer[key], key);

      REQUIRE(ptrTrans.get() == &dummyContainer[key]);
      REQUIRE(ptrTrans2.get() == &dummyContainer[key]);
      REQUIRE(ptrTrans.id() == ProductID());
      REQUIRE(ptrTrans.id() == ptrTrans2.id());
      REQUIRE(ptrTrans.isTransient());
      REQUIRE(ptrTrans2.isTransient());
      REQUIRE(ptrTrans.key() == key);
      REQUIRE(ptrTrans2.key() == key);

      Ptr<Dummy> ptrTrans3(nullptr, key);
      REQUIRE(ptrTrans3.isNull());
    }

    SECTION("set container") {
      unsigned int const key = 2;
      Dummy const dummy;
      DummySet dummyContainer;
      dummyContainer.insert(dummy);
      dummyContainer.insert(dummy);
      dummyContainer.insert(dummy);
      OrphanHandle<DummySet> handle(&dummyContainer, pid);
      Ptr<Dummy> dummyPtr(handle, key);

      REQUIRE(dummyPtr.isAvailable());
      REQUIRE(dummyPtr.id() == pid);
      REQUIRE(dummyPtr.key() == key);
      DummySet::const_iterator it = dummyContainer.begin();
      std::advance(it, key);
      REQUIRE(dummyPtr.get() == it->address());
      REQUIRE(&(*dummyPtr) == it->address());
      REQUIRE((dummyPtr.operator->()) == it->address());
      REQUIRE(dummyPtr->address() == it->address());
    }

    SECTION("list container") {
      unsigned int const key = 2;
      Dummy const dummy;
      DummyList dummyContainer;
      dummyContainer.push_back(dummy);
      dummyContainer.push_back(dummy);
      dummyContainer.push_back(dummy);
      OrphanHandle<DummyList> handle(&dummyContainer, pid);
      Ptr<Dummy> dummyPtr(handle, key);

      REQUIRE(dummyPtr.isAvailable());
      REQUIRE(dummyPtr.id() == pid);
      REQUIRE(dummyPtr.key() == key);
      DummyList::const_iterator it = dummyContainer.begin();
      std::advance(it, key);
      REQUIRE(dummyPtr.get() == it->address());
      REQUIRE(&(*dummyPtr) == it->address());
      REQUIRE((dummyPtr.operator->()) == it->address());
      REQUIRE(dummyPtr->address() == it->address());
    }

    SECTION("deque container") {
      unsigned int const key = 2;
      Dummy const dummy;
      DummyDeque dummyContainer;
      dummyContainer.push_back(dummy);
      dummyContainer.push_back(dummy);
      dummyContainer.push_back(dummy);
      OrphanHandle<DummyDeque> handle(&dummyContainer, pid);
      Ptr<Dummy> dummyPtr(handle, key);

      REQUIRE(dummyPtr.isAvailable());
      REQUIRE(dummyPtr.id() == pid);
      //REQUIRE(dummyPtrProd.id() == pid);
      REQUIRE(dummyPtr.key() == key);
      DummyDeque::const_iterator it = dummyContainer.begin();
      std::advance(it, key);
      REQUIRE(dummyPtr.get() == it->address());
      REQUIRE(&(*dummyPtr) == it->address());
      REQUIRE((dummyPtr.operator->()) == it->address());
      REQUIRE(dummyPtr->address() == it->address());
    }

    SECTION("derived class") {
      unsigned int const key = 2;
      Dummy2 const dummy;
      DummyCollection2 dummyContainer;
      dummyContainer.push_back(dummy);
      dummyContainer.push_back(dummy);
      dummyContainer.push_back(dummy);
      OrphanHandle<DummyCollection2> handle(&dummyContainer, pid);
      Ptr<Dummy> dummyPtr(handle, key);
      Ptr<Dummy2> dummyPtr2(dummyPtr);

      REQUIRE(dummyPtr.isAvailable());
      REQUIRE(dummyPtr.id() == pid);
      REQUIRE(dummyPtr.key() == key);
      REQUIRE(dummyPtr.get() == &dummyContainer[key]);
      REQUIRE(&(*dummyPtr) == &dummyContainer[key]);
      REQUIRE((dummyPtr.operator->()) == &dummyContainer[key]);
      REQUIRE(dummyPtr->address() == dummyContainer[key].address());

      REQUIRE(dummyPtr2.isAvailable());
      REQUIRE(dummyPtr2.id() == pid);
      REQUIRE(dummyPtr2.key() == key);
      REQUIRE(dummyPtr2.get() == &dummyContainer[key]);
      REQUIRE(&(*dummyPtr2) == &dummyContainer[key]);
      REQUIRE((dummyPtr2.operator->()) == &dummyContainer[key]);
      REQUIRE(dummyPtr2->address() == dummyContainer[key].address());

      Ptr<Dummy2> dummy2Ptr(handle, key);
      Ptr<Dummy> copyPtr(dummy2Ptr);
      REQUIRE(dummy2Ptr.key() == copyPtr.key());
      REQUIRE(dummy2Ptr.get() == static_cast<const Dummy*>(dummy2Ptr.get()));

      Ptr<Dummy> copyCtrPtr(copyPtr);
      REQUIRE(copyPtr.key() == copyCtrPtr.key());

      Ptr<Dummy> movePtr(std::move(copyPtr));
      REQUIRE(dummy2Ptr.key() == movePtr.key());
    }
  }

  SECTION("comparison") {
    SECTION("basic comparison") {
      ProductID const pid(1, 1);

      unsigned int const key = 2;
      Dummy const dummy;
      DummyCollection dummyContainer;
      dummyContainer.push_back(dummy);
      OrphanHandle<DummyCollection> handle(&dummyContainer, pid);
      OrphanHandle<DummyCollection> handle2(&dummyContainer, pid);
      Ptr<Dummy> dummyPtr1(handle, key);
      Ptr<Dummy> dummyPtr2(handle2, key);

      REQUIRE(dummyPtr1 == dummyPtr2);
      REQUIRE(!(dummyPtr1 != dummyPtr2));
      REQUIRE(!(dummyPtr1 < dummyPtr2));
      REQUIRE(!(dummyPtr2 < dummyPtr1));
      REQUIRE(!(dummyPtr1 < dummyPtr1));
      REQUIRE(!(dummyPtr2 < dummyPtr2));

      Ptr<Dummy> dummyPtrNewKey(handle, key + 1);
      REQUIRE(!(dummyPtr1 == dummyPtrNewKey));
      REQUIRE(dummyPtr1 != dummyPtrNewKey);
      REQUIRE(dummyPtr1 < dummyPtrNewKey);
      REQUIRE(!(dummyPtrNewKey < dummyPtr1));

      ProductID const pidOther(1, 4);
      OrphanHandle<DummyCollection> handleNewPID(&dummyContainer, pidOther);
      Ptr<Dummy> dummyPtrNewPID(handleNewPID, key);
      REQUIRE(!(dummyPtr1 == dummyPtrNewPID));
      REQUIRE(dummyPtr1 != dummyPtrNewPID);
    }
  }

  SECTION("get") {
    SECTION("IntValue collection") {
      typedef std::vector<IntValue> IntCollection;
      auto ptr = std::make_unique<IntCollection>();

      ptr->push_back(0);
      ptr->push_back(1);

      edm::Wrapper<IntCollection> wrapper(std::move(ptr));
      TestGetter tester;
      tester.hold_ = &wrapper;

      ProductID const pid(1, 1);

      IntCollection const* wptr = dynamic_cast<IntCollection const*>(wrapper.product());

      OrphanHandle<IntCollection> handle(wptr, pid);

      Ptr<IntValue> ref0(pid, 0, &tester);
      REQUIRE(!ref0.hasProductCache());

      Ptr<IntValue> ref1(pid, 1, &tester);

      Ptr<IntValue> ref2(pid, 1, &tester);

      REQUIRE(ref0.isAvailable());
      REQUIRE(0 == ref0->value_);
      REQUIRE(ref0.hasProductCache());
      REQUIRE(1 == ref1->value_);
      REQUIRE(1 == ref2->value_);
      REQUIRE(1 == (*ref1).value_);
    }
    SECTION("IntValue2 collection") {
      typedef std::vector<IntValue2> SDCollection;
      auto ptr = std::make_unique<SDCollection>();

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

      Ptr<IntValue> ref0(pid, 0, &tester);
      REQUIRE(!ref0.hasProductCache());

      Ptr<IntValue> ref1(pid, 1, &tester);

      Ptr<IntValue> ref2(pid, 1, &tester);

      REQUIRE(0 == ref0->value_);
      REQUIRE(ref0.hasProductCache());
      REQUIRE(1 == ref1->value_);
      REQUIRE(1 == ref2->value_);
      REQUIRE(1 == (*ref1).value_);
    }

    SECTION("exception throwing") {
      TestGetter tester;
      tester.hold_ = nullptr;
      ProductID const pid(1, 1);

      Ptr<IntValue> ref0(pid, 0, &tester);
      REQUIRE_THROWS_AS((*ref0), cms::Exception);
      REQUIRE_THROWS_AS((ref0.operator->()), cms::Exception);
    }
  }
}
