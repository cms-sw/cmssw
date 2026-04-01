/*
 *  reftobaseprod_t.cppunit.cc
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
#include "DataFormats/Common/interface/RefToBaseProd.h"

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

  template <typename T, typename C>
  void compareTo(const RefToBaseProd<T>& iProd, C const& iContainer) {
    unsigned int index = 0;
    for (auto const& item : iContainer) {
      REQUIRE(&((*iProd.get())[index++]) == &item);
    }
    index = 0;
    for (auto const& item : iContainer) {
      REQUIRE(&((*iProd)[index++]) == &item);
    }
    index = 0;
    for (auto const& item : iContainer) {
      REQUIRE(&((*(iProd.operator->()))[index++]) == &item);
    }
    index = 0;
    for (auto const& item : iContainer) {
      REQUIRE(&(iProd->at(index++)) == &item);
    }
  }

  struct TestGetter : public edm::EDProductGetter {
    WrapperBase const* hold_;
    WrapperBase const* getIt(ProductID const&) const override { return hold_; }
    unsigned int transitionIndex_() const override { return 0U; }

    TestGetter() : hold_() {}
  };
}  // namespace

TEST_CASE("RefToBaseProd", "[RefToBaseProd]") {
  SECTION("construction") {
    RefToBaseProd<Dummy> nulled;
    REQUIRE(!nulled);
    REQUIRE(nulled.isNull());
    REQUIRE(!nulled.isNonnull());
    REQUIRE(!nulled.isAvailable());

    RefToBaseProd<Dummy> nulledP;
    REQUIRE(!nulledP);
    REQUIRE(nulledP.isNull());
    REQUIRE(!nulledP.isNonnull());
    REQUIRE(!nulledP.isAvailable());

    ProductID const pid(1, 1);

    {
      Dummy const dummy;
      DummyCollection dummyContainer;
      dummyContainer.push_back(dummy);
      dummyContainer.push_back(dummy);
      dummyContainer.push_back(dummy);
      OrphanHandle<DummyCollection> handle(&dummyContainer, pid);
      RefToBaseProd<Dummy> dummyPtr(handle);

      REQUIRE(!dummyPtr.isNull());
      REQUIRE(dummyPtr.isNonnull());
      REQUIRE(dummyPtr.isAvailable());
      REQUIRE(dummyPtr.id() == pid);
      compareTo(dummyPtr, dummyContainer);
    }

    {
      Dummy const dummy;
      DummySet dummyContainer;
      dummyContainer.insert(dummy);
      dummyContainer.insert(dummy);
      dummyContainer.insert(dummy);
      OrphanHandle<DummySet> handle(&dummyContainer, pid);
      RefToBaseProd<Dummy> dummyPtr(handle);

      REQUIRE(dummyPtr.id() == pid);
      compareTo(dummyPtr, dummyContainer);
    }

    {
      Dummy const dummy;
      DummyList dummyContainer;
      dummyContainer.push_back(dummy);
      dummyContainer.push_back(dummy);
      dummyContainer.push_back(dummy);
      OrphanHandle<DummyList> handle(&dummyContainer, pid);
      RefToBaseProd<Dummy> dummyPtr(handle);

      REQUIRE(dummyPtr.id() == pid);
      compareTo(dummyPtr, dummyContainer);
    }

    {
      Dummy const dummy;
      DummyDeque dummyContainer;
      dummyContainer.push_back(dummy);
      dummyContainer.push_back(dummy);
      dummyContainer.push_back(dummy);
      OrphanHandle<DummyDeque> handle(&dummyContainer, pid);
      RefToBaseProd<Dummy> dummyPtr(handle);

      REQUIRE(dummyPtr.id() == pid);
      compareTo(dummyPtr, dummyContainer);
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

      REQUIRE(dummyPtr.id() == pid);
      compareTo(dummyPtr, dummyContainer);

      REQUIRE(dummyPtr2.id() == pid);
      compareTo(dummyPtr2, dummyContainer);

      RefToBaseProd<Dummy> dummyPtr3(std::move(dummyPtr2));
      REQUIRE(dummyPtr3.id() == pid);
      compareTo(dummyPtr3, dummyContainer);
    }
  }

  SECTION("get") {
    {
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

      //NOTE: ROOT will touch the private variables directly and since the RefToBaseProd
      // has no constructor which takes RefCore, I have to play this dirty trick
      assert(sizeof(edm::RefCore) == sizeof(edm::RefToBaseProd<IntValue>));

      RefCore core(pid, nullptr, &tester, false);
      RefToBaseProd<IntValue>& prod = reinterpret_cast<RefToBaseProd<IntValue>&>(core);
      REQUIRE(!prod.isNull());
      REQUIRE(prod.isNonnull());
      REQUIRE(prod.isAvailable());

      //previously making a copy before reading back would cause seg fault
      RefToBaseProd<IntValue> prodCopy(prod);
      REQUIRE(!prodCopy.isNull());
      REQUIRE(prodCopy.isNonnull());
      REQUIRE(prodCopy.isAvailable());

      REQUIRE(!prod.hasCache());

      REQUIRE(0 != prod.get());
      REQUIRE(prod.hasCache());
      compareTo(prod, *wptr);

      REQUIRE(!prodCopy.hasCache());

      REQUIRE(0 != prodCopy.get());
      REQUIRE(prodCopy.hasCache());
      compareTo(prodCopy, *wptr);
    }
    {
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

      //NOTE: ROOT will touch the private variables directly and since the RefToBaseProd
      // has no constructor which takes RefCore, I have to play this dirty trick
      assert(sizeof(edm::RefCore) == sizeof(edm::RefToBaseProd<IntValue>));

      RefCore core(pid, nullptr, &tester, false);
      RefToBaseProd<IntValue>& prod = reinterpret_cast<RefToBaseProd<IntValue>&>(core);

      REQUIRE(!prod.hasCache());

      REQUIRE(0 != prod.get());
      REQUIRE(prod.hasCache());
      compareTo(prod, *wptr);
    }

    {
      TestGetter tester;
      tester.hold_ = nullptr;
      ProductID const pid(1, 1);

      //NOTE: ROOT will touch the private variables directly and since the RefToBaseProd
      // has no constructor which takes RefCore, I have to play this dirty trick
      assert(sizeof(edm::RefCore) == sizeof(edm::RefToBaseProd<IntValue>));

      RefCore core(pid, nullptr, &tester, false);
      RefToBaseProd<IntValue>& prod = reinterpret_cast<RefToBaseProd<IntValue>&>(core);

      REQUIRE_THROWS_AS((*prod), cms::Exception);
      REQUIRE_THROWS_AS((prod.operator->()), cms::Exception);
    }
  }
}
