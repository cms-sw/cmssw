#include <catch2/catch_all.hpp>
#include "DataFormats/Common/interface/CloningPtr.h"

#include <vector>

namespace testcloningptr {
  struct Base {
    virtual ~Base() {}
    virtual int val() const = 0;
    virtual Base* clone() const = 0;
  };

  struct Inherit : public Base {
    Inherit(int iValue) : val_(iValue) {}
    virtual int val() const { return val_; }
    virtual Base* clone() const { return new Inherit(*this); }
    int val_;
  };
}  // namespace testcloningptr

using namespace testcloningptr;

TEST_CASE("test CloningPtr", "[CloningPtr]") {
  using namespace edm;

  Inherit one(1);
  CloningPtr<Base> cpOne(one);
  REQUIRE(&one != cpOne.get());
  REQUIRE(1 == cpOne->val());
  REQUIRE(1 == (*cpOne).val());

  CloningPtr<Base> cpOtherOne(cpOne);
  REQUIRE(cpOne.get() != cpOtherOne.get());
  REQUIRE(cpOtherOne->val() == 1);

  CloningPtr<Base> eqOne;
  eqOne = cpOne;

  REQUIRE(cpOne.get() != eqOne.get());
  REQUIRE(eqOne->val() == 1);
}
