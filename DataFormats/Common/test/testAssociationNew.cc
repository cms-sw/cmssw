#include <catch2/catch_all.hpp>
#include <algorithm>
#include <iterator>
#include <iostream>
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/TestHandle.h"
using namespace edm;

typedef std::vector<double> CVal;
typedef std::vector<int> CKey1;
typedef std::vector<float> CKey2;

namespace {
  struct TestData {
    CVal k;
    CKey1 v1;
    CKey2 v2;
    edm::TestHandle<CVal> handleV;
    edm::TestHandle<CKey1> handleK1;
    edm::TestHandle<CKey2> handleK2;
    std::vector<int> w1, w2;

    void initialize();
    void checkAll();
    void test(const edm::Association<CVal> &);
  };

  void TestData::initialize() {
    k.push_back(1.1);
    k.push_back(2.2);
    k.push_back(3.3);
    ProductID const pidV(1, 1);
    handleV = edm::TestHandle<CVal>(&k, pidV);

    v1.push_back(1);
    v1.push_back(2);
    v1.push_back(3);
    v1.push_back(4);
    ProductID const pidK1(1, 2);
    handleK1 = edm::TestHandle<CKey1>(&v1, pidK1);

    v2.push_back(10.);
    v2.push_back(20.);
    v2.push_back(30.);
    v2.push_back(40.);
    v2.push_back(50.);
    ProductID const pidK2(1, 3);
    handleK2 = edm::TestHandle<CKey2>(&v2, pidK2);

    const int ww1[4] = {2, 1, 0, 2};
    w1.resize(4);
    std::copy(ww1, ww1 + 4, w1.begin());
    const int ww2[5] = {1, 0, 2, 1, -1};
    w2.resize(5);
    std::copy(ww2, ww2 + 5, w2.begin());
  }

  void TestData::checkAll() {
    {
      edm::Association<CVal> assoc(handleV);
      edm::Association<CVal>::Filler filler(assoc);
      filler.insert(handleK1, w1.begin(), w1.end());
      filler.insert(handleK2, w2.begin(), w2.end());
      filler.fill();
      test(assoc);
    }
    {
      edm::Association<CVal> assoc(handleV);
      edm::Association<CVal>::Filler filler1(assoc);
      filler1.insert(handleK1, w1.begin(), w1.end());
      filler1.fill();
      edm::Association<CVal>::Filler filler2(assoc);
      filler2.insert(handleK2, w2.begin(), w2.end());
      filler2.fill();
      test(assoc);
    }
    {
      edm::Association<CVal> assoc1(handleV);
      edm::Association<CVal>::Filler filler1(assoc1);
      filler1.insert(handleK1, w1.begin(), w1.end());
      filler1.fill();
      edm::Association<CVal> assoc2(handleV);
      edm::Association<CVal>::Filler filler2(assoc2);
      filler2.insert(handleK2, w2.begin(), w2.end());
      filler2.fill();
      edm::Association<CVal> assoc = assoc1 + assoc2;
      test(assoc);
    }
  }

  void TestData::test(const edm::Association<CVal> &assoc) {
    REQUIRE(!assoc.contains(ProductID(1, 1)));
    REQUIRE(assoc.contains(ProductID(1, 2)));
    REQUIRE(assoc.contains(ProductID(1, 3)));
    REQUIRE(!assoc.contains(ProductID(1, 4)));
    edm::Ref<CVal> r1 = assoc[edm::Ref<CKey1>(handleK1, 0)];
    edm::Ref<CVal> r2 = assoc[edm::Ref<CKey1>(handleK1, 1)];
    edm::Ref<CVal> r3 = assoc[edm::Ref<CKey1>(handleK1, 2)];
    edm::Ref<CVal> r4 = assoc[edm::Ref<CKey1>(handleK1, 3)];
    REQUIRE(r1.isNonnull());
    REQUIRE(r2.isNonnull());
    REQUIRE(r3.isNonnull());
    REQUIRE(r4.isNonnull());
    REQUIRE(*r1 == k[w1[0]]);
    REQUIRE(*r2 == k[w1[1]]);
    REQUIRE(*r3 == k[w1[2]]);
    REQUIRE(*r4 == k[w1[3]]);
    edm::Ref<CVal> s1 = assoc[edm::Ref<CKey2>(handleK2, 0)];
    edm::Ref<CVal> s2 = assoc[edm::Ref<CKey2>(handleK2, 1)];
    edm::Ref<CVal> s3 = assoc[edm::Ref<CKey2>(handleK2, 2)];
    edm::Ref<CVal> s4 = assoc[edm::Ref<CKey2>(handleK2, 3)];
    edm::Ref<CVal> s5 = assoc[edm::Ref<CKey2>(handleK2, 4)];
    REQUIRE(s1.isNonnull());
    REQUIRE(s2.isNonnull());
    REQUIRE(s3.isNonnull());
    REQUIRE(s4.isNonnull());
    REQUIRE(s5.isNull());
    REQUIRE(*s1 == k[w2[0]]);
    REQUIRE(*s2 == k[w2[1]]);
    REQUIRE(*s3 == k[w2[2]]);
    REQUIRE(*s4 == k[w2[3]]);
    REQUIRE(assoc.size() == w1.size() + w2.size());
  }
}  // namespace

TEST_CASE("AssociationNew", "[AssociationNew]") {
  TestData data;
  data.initialize();

  SECTION("checkAll") { data.checkAll(); }
}
