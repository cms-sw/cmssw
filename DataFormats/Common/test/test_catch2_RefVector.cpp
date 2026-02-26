#include <vector>
#include "catch2/catch_all.hpp"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

TEST_CASE("RefVector", "[RefVector]") {
  SECTION("iteration") {
    using product_t = std::vector<double>;
    using ref_t = edm::Ref<product_t>;
    using refvec_t = edm::RefVector<product_t>;

    product_t product;
    product.push_back(1.0);
    product.push_back(100.0);
    product.push_back(0.5);
    product.push_back(2.0);

    refvec_t refvec;
    REQUIRE(refvec.size() == 0);
    REQUIRE(refvec.empty());

    ref_t ref0(edm::ProductID(1, 1), &product[0], 0);
    refvec.push_back(ref0);

    ref_t ref1(edm::ProductID(1, 1), &product[2], 2);
    refvec.push_back(ref1);

    ref_t ref2(edm::ProductID(1, 1), &product[3], 3);
    refvec.push_back(ref2);

    auto iter = refvec.begin();
    REQUIRE(iter->id() == edm::ProductID(1, 1));
    REQUIRE(iter->key() == 0);
    REQUIRE(*(iter->get()) == 1.0);
    ++iter;

    REQUIRE(iter->id() == edm::ProductID(1, 1));
    REQUIRE(iter->key() == 2);
    REQUIRE(*(iter->get()) == 0.5);
    ++iter;

    REQUIRE(iter->id() == edm::ProductID(1, 1));
    REQUIRE(iter->key() == 3);
    REQUIRE(*(iter->get()) == 2.0);
    ++iter;

    REQUIRE(iter == refvec.end());
  }
}
