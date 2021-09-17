#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/ThinnedRefSet.h"

#include "SimpleEDProductGetter.h"

#include <memory>
#include <vector>

TEST_CASE("ThinnedRefSet", "[ThinnedRefSet]") {
  SECTION("Default constructor") { edm::ThinnedRefSet<int> set; }

  SECTION("std::vector<int>") {
    SimpleEDProductGetter getter;

    edm::ProductID id(1, 42U);
    REQUIRE(id.isValid());

    auto prod = std::make_unique<std::vector<int>>();
    for (int i = 0; i < 10; ++i) {
      prod->push_back(i);
    }
    getter.addProduct(id, std::move(prod));

    edm::ThinnedRefSet<std::vector<int>> refSet;
    edm::RefProd<std::vector<int>> refProd(id, &getter);
    auto filler = refSet.fill(refProd, getter);

    filler.insert(edm::Ref<std::vector<int>>(id, 0, &getter));
    filler.insert(edm::Ref<std::vector<int>>(id, 4, &getter));
    filler.insert(edm::Ref<std::vector<int>>(id, 9, &getter));

    REQUIRE(refSet.contains(0));
    REQUIRE(not refSet.contains(1));
    REQUIRE(not refSet.contains(2));
    REQUIRE(not refSet.contains(3));
    REQUIRE(refSet.contains(4));
    REQUIRE(not refSet.contains(5));
    REQUIRE(not refSet.contains(6));
    REQUIRE(not refSet.contains(7));
    REQUIRE(not refSet.contains(8));
    REQUIRE(refSet.contains(9));
    REQUIRE(not refSet.contains(10));
  }

  SECTION("edmNew::DetSetVector<int>") {
    SimpleEDProductGetter getter;

    edm::ProductID id(1, 75U);
    REQUIRE(id.isValid());

    {
      auto detsets = std::make_unique<edmNew::DetSetVector<int>>(2);
      {
        edmNew::DetSetVector<int>::FastFiller filler(*detsets, 1);
        filler.push_back(1);
        filler.push_back(2);
        filler.push_back(3);
      }
      {
        edmNew::DetSetVector<int>::FastFiller filler(*detsets, 2);
        filler.push_back(10);
        filler.push_back(20);
      }
      REQUIRE(detsets->size() == 2);
      REQUIRE(detsets->dataSize() == 5);
      REQUIRE(detsets->detsetSize(0) == 3);
      REQUIRE(detsets->detsetSize(1) == 2);

      auto ds = detsets->find(1);
      REQUIRE(ds != detsets->end());
      REQUIRE(ds->id() == 1);
      REQUIRE(detsets->find(3) == detsets->end());

      getter.addProduct(id, std::move(detsets));
    }

    edm::ThinnedRefSet<edmNew::DetSetVector<int>> refSet;
    edm::RefProd<edmNew::DetSetVector<int>> refProd(id, &getter);
    auto filler = refSet.fill(refProd, getter);

    REQUIRE(*(edm::Ref<edmNew::DetSetVector<int>, int>(id, 0, &getter)) == 1);
    REQUIRE(*(edm::Ref<edmNew::DetSetVector<int>, int>(id, 2, &getter)) == 3);
    REQUIRE(*(edm::Ref<edmNew::DetSetVector<int>, int>(id, 4, &getter)) == 20);

    filler.insert(edm::Ref<edmNew::DetSetVector<int>, int>(id, 0, &getter));
    filler.insert(edm::Ref<edmNew::DetSetVector<int>, int>(id, 2, &getter));
    filler.insert(edm::Ref<edmNew::DetSetVector<int>, int>(id, 4, &getter));

    REQUIRE(refSet.contains(0));
    REQUIRE(not refSet.contains(1));
    REQUIRE(refSet.contains(2));
    REQUIRE(not refSet.contains(3));
    REQUIRE(refSet.contains(4));
    REQUIRE(not refSet.contains(5));
  }
}
