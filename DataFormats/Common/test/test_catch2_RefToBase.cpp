#include "catch2/catch_all.hpp"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/TestHandle.h"

#include <vector>

namespace testreftobase {
  struct Base {
    virtual ~Base() {}
    virtual int val() const = 0;
  };

  struct Inherit1 : public Base {
    virtual int val() const { return 1; }
  };
  struct Inherit2 : public Base {
    virtual int val() const { return 2; }
  };
}  // namespace testreftobase

using namespace testreftobase;

TEST_CASE("test RefToBase", "[RefToBase]") {
  SECTION("Class type") {
    using namespace edm;

    std::vector<Inherit1> v1(2, Inherit1());

    TestHandle<std::vector<Inherit1> > h1(&v1, ProductID(1, 1));
    Ref<std::vector<Inherit1> > r1(h1, 1);
    RefToBase<Base> b1(r1);

    SECTION("Value check") {
      CHECK(&(*b1) == static_cast<Base*>(&(v1[1])));
      CHECK(b1.operator->() == b1.get());
      CHECK(b1.get() == static_cast<Base*>(&(v1[1])));
      CHECK(b1.id() == ProductID(1, 1));
    }
    SECTION("Copy constructor") {
      //copy constructor
      RefToBase<Base> b2(b1);
      CHECK(&(*b2) == static_cast<Base*>(&(v1[1])));
      CHECK(b2.id() == b1.id());
    }

    SECTION("operator=") {
      //operator=
      RefToBase<Base> b3;
      CHECK(b3.isNull());
      CHECK(!(b3.isNonnull()));
      CHECK(!b3);
      b3 = b1;
      CHECK(&(*b3) == static_cast<Base*>(&(v1[1])));
      CHECK(b3.id() == b1.id());
      CHECK(!(b3.isNull()));
      CHECK(b3.isNonnull());
      CHECK(!(!b3));
    }

    SECTION("castTo inheriting type") { CHECK(b1.castTo<Ref<std::vector<Inherit1> > >() == r1); }
    SECTION("castTo incorrect inherting type") {
      CHECK_THROWS_AS(b1.castTo<Ref<std::vector<Inherit2> > >(), cms::Exception);
    }
    /*
    Uncomment to test compile time failure
    SECTION("castTo unrelated type") {
      b1.castTo<Ref<std::vector<std::string> > >();
    } */
  }

  SECTION("builtin type") {
    using namespace edm;
    std::vector<int> v1(2, 3);

    TestHandle<std::vector<int> > h1(&v1, ProductID(1, 1));
    Ref<std::vector<int> > r1(h1, 1);
    RefToBase<int> b1(r1);

    SECTION("Value check") {
      CHECK(&(*b1) == &v1[1]);
      CHECK(b1.operator->() == b1.get());
      CHECK(b1.get() == (&(v1[1])));
      CHECK(b1.id() == ProductID(1, 1));
    }
    SECTION("castTo Ref") { CHECK(b1.castTo<Ref<std::vector<int> > >() == r1); }
    /* uncomment to test compile time failure
    SECTION("bad castTo Ref") { b1.castTo<Ref<std::vector<double> > >(); }
    */
  }
}
