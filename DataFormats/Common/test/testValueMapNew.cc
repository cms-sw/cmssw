#include <catch2/catch_all.hpp>
#include <algorithm>
#include <iterator>
#include <iostream>
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/TestHandle.h"
using namespace edm;

typedef std::vector<int> CKey1;
typedef std::vector<float> CKey2;

TEST_CASE("test ValueMap", "[ValueMap]") {
  // Initialize test data
  CKey1 v1;
  v1.push_back(1);
  v1.push_back(2);
  v1.push_back(3);
  v1.push_back(4);
  ProductID const pidK1(1, 2);
  edm::TestHandle<CKey1> handleK1(&v1, pidK1);

  CKey2 v2;
  v2.push_back(10.);
  v2.push_back(20.);
  v2.push_back(30.);
  v2.push_back(40.);
  v2.push_back(50.);
  ProductID const pidK2(1, 3);
  edm::TestHandle<CKey2> handleK2(&v2, pidK2);

  std::vector<int> w1;
  const int ww1[4] = {2, 1, 0, 2};
  w1.resize(4);
  std::copy(ww1, ww1 + 4, w1.begin());

  std::vector<int> w2;
  const int ww2[5] = {1, 0, 2, 1, -1};
  w2.resize(5);
  std::copy(ww2, ww2 + 5, w2.begin());

  // Helper lambda to verify ValueMap contents
  auto test = [&](const edm::ValueMap<int> &values) {
    REQUIRE(values.idSize() == 2);
    REQUIRE(!values.contains(ProductID(1, 0)));
    REQUIRE(!values.contains(ProductID(1, 1)));
    REQUIRE(values.contains(ProductID(1, 2)));
    REQUIRE(values.contains(ProductID(1, 3)));
    REQUIRE(!values.contains(ProductID(1, 4)));
    REQUIRE(!values.contains(ProductID(1, 5)));

    int r1 = values[edm::Ref<CKey1>(handleK1, 0)];
    int r2 = values[edm::Ref<CKey1>(handleK1, 1)];
    int r3 = values[edm::Ref<CKey1>(handleK1, 2)];
    int r4 = values[edm::Ref<CKey1>(handleK1, 3)];
    REQUIRE(r1 == w1[0]);
    REQUIRE(r2 == w1[1]);
    REQUIRE(r3 == w1[2]);
    REQUIRE(r4 == w1[3]);

    int s1 = values[edm::Ref<CKey2>(handleK2, 0)];
    int s2 = values[edm::Ref<CKey2>(handleK2, 1)];
    int s3 = values[edm::Ref<CKey2>(handleK2, 2)];
    int s4 = values[edm::Ref<CKey2>(handleK2, 3)];
    int s5 = values[edm::Ref<CKey2>(handleK2, 4)];
    REQUIRE(s1 == w2[0]);
    REQUIRE(s2 == w2[1]);
    REQUIRE(s3 == w2[2]);
    REQUIRE(s4 == w2[3]);
    REQUIRE(s5 == w2[4]);

    REQUIRE(values.size() == w1.size() + w2.size());

    edm::ValueMap<int>::const_iterator b = values.begin(), e = values.end(), i;
    REQUIRE(e - b == 2);
    REQUIRE(b.id() == ProductID(1, 2));
    REQUIRE((b + 1).id() == ProductID(1, 3));

    ProductID pids[] = {ProductID(1, 2), ProductID(1, 3)};
    const std::vector<int> *w[] = {&w1, &w2};
    for (i = b; i != e; ++i) {
      size_t idx = i - b;
      REQUIRE(i.id() == pids[idx]);
      REQUIRE(i.size() == w[idx]->size());
      {
        std::vector<int>::const_iterator bb = i.begin(), ee = i.end(), j;
        for (j = bb; j != ee; ++j) {
          size_t jdx = j - bb;
          REQUIRE(*j == (*w[idx])[jdx]);
          REQUIRE(i[jdx] == (*w[idx])[jdx]);
        }
      }
    }
  };

  // Test scenario 1: Insert both collections with a single filler
  {
    edm::ValueMap<int> values;
    edm::ValueMap<int>::Filler filler(values);
    filler.insert(handleK1, w1.begin(), w1.end());
    filler.insert(handleK2, w2.begin(), w2.end());
    filler.fill();
    test(values);
  }

  // Test scenario 2: Insert each collection with separate fillers
  {
    edm::ValueMap<int> values;
    edm::ValueMap<int>::Filler filler1(values);
    filler1.insert(handleK1, w1.begin(), w1.end());
    filler1.fill();
    edm::ValueMap<int>::Filler filler2(values);
    filler2.insert(handleK2, w2.begin(), w2.end());
    filler2.fill();
    test(values);
  }

  // Test scenario 3: Create two separate ValueMaps and combine them
  {
    edm::ValueMap<int> values1;
    edm::ValueMap<int>::Filler filler1(values1);
    filler1.insert(handleK1, w1.begin(), w1.end());
    filler1.fill();

    edm::ValueMap<int> values2;
    edm::ValueMap<int>::Filler filler2(values2);
    filler2.insert(handleK2, w2.begin(), w2.end());
    filler2.fill();

    edm::ValueMap<int> values = values1 + values2;
    test(values);
  }
}
