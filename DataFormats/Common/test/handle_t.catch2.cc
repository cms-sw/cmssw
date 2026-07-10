#include <catch2/catch_all.hpp>
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ValidHandle.h"
#include "DataFormats/Common/interface/FunctorHandleExceptionFactory.h"
#include "FWCore/Utilities/interface/Exception.h"

TEST_CASE("test Handle", "[Handle]") {
  using namespace edm;

  {
    Handle<int> hDefault;
    REQUIRE(not hDefault.isValid());
    REQUIRE(not hDefault.failedToGet());

    //the following leads to a seg fault :(
    //REQUIRE(hDefault.id() == ProductID{});
    REQUIRE(not hDefault.whyFailed());

    //This doesn't throw
    //REQUIRE_THROWS_AS([&hDefault](){*hDefault;},
    //                     cms::Exception);
  }

  {
    Provenance provDummy;
    int value = 3;

    Handle<int> h(&value, &provDummy);
    REQUIRE(h.isValid());
    REQUIRE(not h.failedToGet());
    REQUIRE(not h.whyFailed());
    REQUIRE(3 == *h);

    Handle<int> hCopy(h);
    REQUIRE(hCopy.isValid());
    REQUIRE(not hCopy.failedToGet());
    REQUIRE(not hCopy.whyFailed());
    REQUIRE(3 == *hCopy);

    Handle<int> hOpEq;
    hOpEq = h;
    REQUIRE(hOpEq.isValid());
    REQUIRE(not hOpEq.failedToGet());
    REQUIRE(not hOpEq.whyFailed());
    REQUIRE(3 == *hOpEq);

    Handle<int> hOpEqMove;
    hOpEqMove = std::move(hCopy);
    REQUIRE(hOpEqMove.isValid());
    REQUIRE(not hOpEqMove.failedToGet());
    REQUIRE(not hOpEqMove.whyFailed());
    REQUIRE(3 == *hOpEqMove);
  }

  {
    Handle<int> hFail(makeHandleExceptionFactory(
        []() -> std::shared_ptr<cms::Exception> { return std::make_shared<cms::Exception>("DUMMY"); }));

    REQUIRE(not hFail.isValid());
    REQUIRE(hFail.failedToGet());

    REQUIRE(hFail.whyFailed());

    REQUIRE_THROWS_AS(*hFail, cms::Exception);
  }
}

TEST_CASE("test ValidHandle", "[Handle]") {
  using namespace edm;

  {
    ProductID dummyID;
    int value = 3;

    ValidHandle<int> h(&value, dummyID);

    REQUIRE(h.product() == &value);
    REQUIRE(*h == value);
    REQUIRE(h.operator->() == &value);
    REQUIRE(dummyID == h.id());
  }

  {
    ProductID dummyID;
    REQUIRE_THROWS_AS(ValidHandle<int>(nullptr, dummyID), cms::Exception);
  }
  {
    Handle<int> hDefault;
    REQUIRE_THROWS_AS(makeValid(hDefault), cms::Exception);
    //This doesn't throw
    //REQUIRE_THROWS_AS([&hDefault](){*hDefault;},
    //                     cms::Exception);
  }

  {
    Provenance provDummy;
    int value = 3;

    Handle<int> h(&value, &provDummy);

    auto hv = makeValid(h);

    REQUIRE(3 == *hv);
    REQUIRE(h.id() == hv.id());

    ValidHandle<int> hCopy(hv);
    REQUIRE(3 == *hCopy);

    int value2 = 2;
    ValidHandle<int> hOpEq(&value2, provDummy.productID());
    REQUIRE(2 == *hOpEq);
    hOpEq = ValidHandle<int>(value, provDummy.productID());
  }

  {
    Handle<int> hFail(makeHandleExceptionFactory(
        []() -> std::shared_ptr<cms::Exception> { return std::make_shared<cms::Exception>("DUMMY"); }));

    REQUIRE_THROWS_AS(makeValid(hFail), cms::Exception);
  }
}
