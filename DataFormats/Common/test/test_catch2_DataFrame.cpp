#include <vector>
#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <cstring>
#include <random>
#include "catch2/catch_all.hpp"
#include "DataFormats/Common/interface/DataFrame.h"
#include "DataFormats/Common/interface/DataFrameContainer.h"

namespace {
  struct VerifyIter {
    VerifyIter(const std::vector<edm::DataFrame::data_type>& sv1, const std::vector<edm::DataFrame::data_type>& sv2)
        : n(0), sv1_(sv1), sv2_(sv2) {}
    void operator()(edm::DataFrame const& df) {
      ++n;
      REQUIRE(df.id() == 2000 + n);
      std::vector<edm::DataFrame::data_type> v2(10);
      std::copy(df.begin(), df.end(), v2.begin());
      if (n % 2 == 0)
        REQUIRE(sv1_ == v2);
      else
        REQUIRE(sv2_ == v2);
    }
    unsigned int n;
    const std::vector<edm::DataFrame::data_type>& sv1_;
    const std::vector<edm::DataFrame::data_type>& sv2_;
  };
}  // namespace

class TestDataFrame {
public:
  static auto& get(edm::DataFrameContainer& iC) { return iC.m_data; }
  static auto& get(edm::DataFrame& iC) { return iC.m_data; }
};

TEST_CASE("DataFrameContainer and DataFrame", "[DataFrame]") {
  std::vector<edm::DataFrame::data_type> sv1(10), sv2(10);
  edm::DataFrame::data_type v[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::copy(v, v + 10, sv1.begin());
  std::transform(sv1.begin(), sv1.end(), sv2.begin(), [](auto x) { return 10 + x; });

  SECTION("default ctor") {
    edm::DataFrameContainer frames(10, 2);
    REQUIRE(frames.stride() == 10);
    REQUIRE(frames.subdetId() == 2);
    REQUIRE(frames.size() == 0);
    frames.resize(3);
    REQUIRE(frames.size() == 3);
    edm::DataFrame df = frames[1];
    REQUIRE(df.size() == 10);
    REQUIRE(TestDataFrame::get(df) == &TestDataFrame::get(frames).front() + 10);
    df.set(frames, 2);
    REQUIRE(df.size() == 10);
    REQUIRE(TestDataFrame::get(df) == &TestDataFrame::get(frames).front() + 20);
    frames.pop_back();
    REQUIRE(frames.size() == 2);
    REQUIRE(TestDataFrame::get(frames).size() == 20);
  }

  SECTION("filling") {
    edm::DataFrameContainer frames(10, 2);
    for (unsigned int n = 1; n < 5; ++n) {
      unsigned int id = 20 + n;
      frames.push_back(id);
      REQUIRE(frames.size() == n);
      edm::DataFrame df = frames.back();
      REQUIRE(df.size() == 10);
      REQUIRE(df.id() == id);
      if (n % 2 == 0)
        std::copy(sv1.begin(), sv1.end(), df.begin());
      else
        ::memcpy(&df[0], &sv1[0], sizeof(edm::DataFrame::data_type) * frames.stride());
      std::vector<edm::DataFrame::data_type> v2(10);
      std::copy(
          TestDataFrame::get(frames).begin() + (n - 1) * 10, TestDataFrame::get(frames).begin() + n * 10, v2.begin());
      REQUIRE(sv1 == v2);
    }
  }

  SECTION("iterator") {
    edm::DataFrameContainer frames(10, 2);
    for (int n = 1; n < 5; ++n) {
      int id = 2000 + n;
      frames.push_back(id);
      edm::DataFrame df = frames.back();
      if (n % 2 == 0)
        std::copy(sv1.begin(), sv1.end(), df.begin());
      else
        std::copy(sv2.begin(), sv2.end(), df.begin());
    }
    VerifyIter verify(sv1, sv2);
    REQUIRE(std::for_each(frames.begin(), frames.end(), verify).n == 4);
  }

  SECTION("sort") {
    edm::DataFrameContainer frames(10, 2);
    std::vector<unsigned int> ids(100, 1);
    ids[0] = 2001;
    std::partial_sum(ids.begin(), ids.end(), ids.begin());
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(ids.begin(), ids.end(), g);
    for (int n = 0; n < 100; ++n) {
      frames.push_back(ids[n]);
      edm::DataFrame df = frames.back();
      if (ids[n] % 2 == 0)
        std::copy(sv1.begin(), sv1.end(), df.begin());
      else
        std::copy(sv2.begin(), sv2.end(), df.begin());
    }
    frames.sort();
    VerifyIter verify(sv1, sv2);
    REQUIRE(std::for_each(frames.begin(), frames.end(), verify).n == 100);
  }
}
