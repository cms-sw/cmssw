#include <catch.hpp>

#include "DataFormats/Common/interface/TrivialCopyTraits.h"

#include <cmath>
#include <map>
#include <type_traits>
#include <vector>
#include <numeric>
#include <cstring>

// Catch2 tests for TrivialCopyTraits
//
// This test file defines various types, and specializes TrivialCopyTraits for them.
//
// The following tests are performed:
//
// - TrivialCopyTraits specializations for int and double are correct. In
// particular, their memory regions are correct.
// - TrivialCopyTraits specialization for std::vector<float> is correct. In
// particular, one can successfully initialize a vector from the properties of
// another.
// - A memcpy-able struct "S" can be copied correctly using its TrivialCopyTraits
// specialization, and the method edm::TrivialCopyTraits<S>::finalize() works
// correctly.
// - It can be checked that there is no TrivialCopyTraits specialization for
// std::map.
// - A type "S2" whose specialization requires initialization but has no
// properties can be copied correctly through the TrivialCopyTraits interface.
// - Several bad TrivialCopyTraits specializations can be detected correctly at
// compile time.
//

// --------------------------------------------------------------------------
// Definitions of various types, and their TrivialCopyTraits specializations

// A trivially copyable type and its nice TrivialCopyTraits specialization
struct S {
  S() = default;
  S(std::string m, std::vector<float> v) : msg{std::move(m)}, vec{std::move(v)} { setVecSum(); }

  std::string msg;
  std::vector<float> vec;
  float vec_sum = 0.0f;  // something that needs to be calculated after the copy is finished
  void setVecSum() { vec_sum = std::accumulate(vec.begin(), vec.end(), 0.0f); }
};

template <>
struct edm::TrivialCopyTraits<S> {
  using Properties = std::array<size_t, 2>;  // {vec.size(), s.size()}

  static Properties properties(S const& object) {
    return std::array<size_t, 2>{{object.vec.size(), object.msg.size()}};
  }

  static void initialize(S& object, Properties const& sizes) {
    object.vec.resize(sizes.at(0));
    object.msg.resize(sizes.at(1));
  }

  static void finalize(S& object) { object.setVecSum(); }

  static std::vector<std::span<std::byte>> regions(S& object) {
    return {{reinterpret_cast<std::byte*>(object.msg.data()), object.msg.size()},
            {reinterpret_cast<std::byte*>(object.vec.data()), object.vec.size() * sizeof(float)}};
  }

  static std::vector<std::span<const std::byte>> regions(S const& object) {
    return {{reinterpret_cast<std::byte const*>(object.msg.data()), object.msg.size()},
            {reinterpret_cast<std::byte const*>(object.vec.data()), object.vec.size() * sizeof(float)}};
  }
};

// A type that does not have properties but requires initialization, and its valid
// TrivialCopyTraits specialization
struct S2 {
  S2() { vec.resize(size); }

  // vec requires initialization, but doesn't need properties because its size is
  // fixed
  const size_t size = 10;
  std::vector<int> vec;
};

template <>
struct edm::TrivialCopyTraits<S2> {
  static void initialize(S2& object) { object.vec.resize(object.size); }

  static std::vector<std::span<std::byte>> regions(S2& object) {
    return {{reinterpret_cast<std::byte*>(object.vec.data()), object.vec.size() * sizeof(int)}};
  }

  static std::vector<std::span<const std::byte>> regions(S2 const& object) {
    return {{reinterpret_cast<std::byte const*>(object.vec.data()), object.vec.size() * sizeof(int)}};
  }
};

// A bad TrivialCopyTraits specialization, that is missing regions()
struct S3 {
  std::vector<int> vec;
};

template <>
struct edm::TrivialCopyTraits<S3> {
  static void initialize(S3& object) { object.vec.resize(10); }
};

// A bad TrivialCopyTraits specialization, with a non-valid initialize()
struct S4 {
  std::vector<int> data;
};

template <>
struct edm::TrivialCopyTraits<S4> {
  using Properties = size_t;

  static Properties properties(S4 const& object) { return object.data.size(); }

  // initialize should take two arguments; the object and its properties (size)
  static void initialize(S4& object) { object.data.resize(5); }

  static std::vector<std::span<std::byte>> regions(S4& object) {
    return {{reinterpret_cast<std::byte*>(object.data.data()), object.data.size() * sizeof(int)}};
  }

  static std::vector<std::span<const std::byte>> regions(S4 const& object) {
    return {{reinterpret_cast<std::byte const*>(object.data.data()), object.data.size() * sizeof(int)}};
  }
};

// A bad TrivialCopyTraits specialization, missing only the const regions
struct S5 {
  int value;
};

template <>
struct edm::TrivialCopyTraits<S5> {
  static std::vector<std::span<std::byte>> regions(S5& object) {
    return {{reinterpret_cast<std::byte*>(&object.value), sizeof(int)}};
  }
};

// --------------------------------------------------------------------------
// The tests

TEST_CASE("test TrivialCopyTraits", "[TrivialCopyTraits]") {
  SECTION("int") {
    REQUIRE(edm::HasTrivialCopyTraits<int>);

    auto checkInt = [](int v) {
      // test non-const regions
      auto regions = edm::TrivialCopyTraits<int>::regions(v);
      REQUIRE(regions.size() == 1);
      REQUIRE(regions[0].size() == sizeof(int));
      REQUIRE(regions[0].data() == reinterpret_cast<std::byte*>(&v));

      // test const regions
      const int const_v = v;
      auto const_regions = edm::TrivialCopyTraits<int>::regions(const_v);
      REQUIRE(const_regions.size() == 1);
      REQUIRE(const_regions[0].size() == sizeof(int));
      REQUIRE(const_regions[0].data() == reinterpret_cast<const std::byte*>(&const_v));
    };

    checkInt(-1);
    checkInt(42);
    checkInt(std::numeric_limits<int>::max());
    checkInt(std::numeric_limits<int>::min());
  }

  SECTION("double") {
    REQUIRE(edm::HasTrivialCopyTraits<double>);

    auto checkDouble = [](double v) {
      // test non-const regions
      auto regions = edm::TrivialCopyTraits<double>::regions(v);
      REQUIRE(regions.size() == 1);
      REQUIRE(regions[0].size() == sizeof(double));
      REQUIRE(regions[0].data() == reinterpret_cast<std::byte*>(&v));

      // test const regions
      const double const_v = v;
      auto const_regions = edm::TrivialCopyTraits<double>::regions(const_v);
      REQUIRE(const_regions.size() == 1);
      REQUIRE(const_regions[0].size() == sizeof(double));
      REQUIRE(const_regions[0].data() == reinterpret_cast<const std::byte*>(&const_v));
    };

    checkDouble(-1.0);
    checkDouble(std::sqrt(2.));
    checkDouble(std::numeric_limits<double>::max());
    checkDouble(std::numeric_limits<double>::min());
    checkDouble(std::numeric_limits<double>::epsilon());
  }

  SECTION("std::vector<float>") {
    using VectorType = std::vector<float>;

    REQUIRE(edm::HasTrivialCopyTraits<VectorType>);
    REQUIRE(edm::HasTrivialCopyProperties<VectorType>);
    REQUIRE(edm::HasValidInitialize<VectorType>);

    VectorType vec = {-5.5f, -3.3f, -1.1f, 4.4f, 8.8f};

    // Test properties
    auto vec_size = edm::TrivialCopyTraits<VectorType>::properties(vec);
    REQUIRE(vec_size == 5);

    // Test initialize
    VectorType new_vec;
    edm::TrivialCopyTraits<VectorType>::initialize(new_vec, vec_size);
    REQUIRE(new_vec.size() == 5);

    // Test non-const regions
    auto regions = edm::TrivialCopyTraits<VectorType>::regions(vec);
    REQUIRE(regions.size() == 1);
    REQUIRE(regions[0].size() == vec.size() * sizeof(float));
    REQUIRE(regions[0].data() == reinterpret_cast<std::byte*>(vec.data()));

    // Test const regions
    const VectorType const_vec = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f};
    auto const_regions = edm::TrivialCopyTraits<VectorType>::regions(const_vec);
    REQUIRE(const_regions.size() == 1);
    REQUIRE(const_regions[0].size() == const_vec.size() * sizeof(float));
    REQUIRE(const_regions[0].data() == reinterpret_cast<const std::byte*>(const_vec.data()));
  }

  SECTION("memcpy-able struct") {
    std::string test_msg = "hello!";
    std::vector<float> test_vec = {-1.0f, 4.0f, 42.0f};
    float test_vec_sum = std::accumulate(test_vec.begin(), test_vec.end(), 0.0f);

    // initialize a memcpy-able struct s
    REQUIRE(edm::HasTrivialCopyTraits<S>);
    REQUIRE(edm::HasTrivialCopyProperties<S>);
    REQUIRE(edm::HasValidInitialize<S>);
    S s{test_msg, test_vec};

    // initialize a clone of s
    S s_clone;
    edm::TrivialCopyTraits<S>::initialize(s_clone, edm::TrivialCopyTraits<S>::properties(s));

    // Get memory regions
    auto const s_regions = edm::TrivialCopyTraits<S>::regions(s);
    auto s_clone_regions = edm::TrivialCopyTraits<S>::regions(s_clone);

    REQUIRE(s_regions.size() == s_clone_regions.size());
    REQUIRE(s_clone.msg.size() == s.msg.size());
    REQUIRE(s_clone.vec.size() == s.vec.size());

    for (size_t i = 0; i < s_regions.size(); ++i) {
      // check that initialize worked, i.e. enough memory in s_clone has been made available to copy s into it
      REQUIRE(s_regions.at(i).size_bytes() == s_clone_regions.at(i).size_bytes());

      // do the copy
      std::memcpy(s_clone_regions.at(i).data(), s_regions.at(i).data(), s_regions.at(i).size_bytes());
    }

    //s_clone.vec_sum has not been touched yet
    REQUIRE(s_clone.vec_sum == 0.0f);

    // finalize the clone, which should calculate vec_sum
    edm::TrivialCopyTraits<S>::finalize(s_clone);

    // check that the copy worked
    REQUIRE(s_clone.vec == test_vec);
    REQUIRE(s_clone.msg == test_msg);

    // check that finalize worked
    REQUIRE(s_clone.vec_sum == test_vec_sum);
  }

  SECTION("std::map<int,int>") {
    using MapType = std::map<int, int>;

    // there is no TrivialCopyTraits specialization for std::map (and there shouldn't be, since std::map is not trivially copyable)
    static_assert(!edm::HasTrivialCopyTraits<MapType>);
  }

  SECTION("A valid specialization with initialize() but without properties()") {
    REQUIRE(edm::HasTrivialCopyTraits<S2>);
    REQUIRE(!edm::HasTrivialCopyProperties<S2>);
    REQUIRE(edm::HasValidInitialize<S2>);

    S2 s2;
    // fill its member vector with some data
    for (size_t i = 0; i < s2.vec.size(); ++i) {
      s2.vec[i] = static_cast<int>(i * 10);
    }

    S2 s2_clone;
    // initialize the clone (no properties required)
    edm::TrivialCopyTraits<S2>::initialize(s2_clone);

    // get memory regions
    auto const s2_regions = edm::TrivialCopyTraits<S2>::regions(s2);
    auto s2_clone_regions = edm::TrivialCopyTraits<S2>::regions(s2_clone);

    // Only one memory region (the vector)
    REQUIRE(s2_regions.size() == 1);
    REQUIRE(s2_clone_regions.size() == 1);
    REQUIRE(s2_regions[0].size_bytes() == s2_clone_regions[0].size_bytes());

    // before the copy:
    REQUIRE(s2_clone.vec != s2.vec);

    // do the copy
    std::memcpy(s2_clone_regions[0].data(), s2_regions[0].data(), s2_regions[0].size_bytes());

    // and now,
    REQUIRE(s2_clone.vec == s2.vec);
  }

  SECTION("Invalid specializations") {
    // S3: Missing regions() method
    static_assert(!edm::HasTrivialCopyTraits<S3>);
    static_assert(!edm::HasRegions<S3>);

    // S5: Missing just the const regions() overload
    static_assert(!edm::HasTrivialCopyTraits<S5>);
    static_assert(!edm::HasRegions<S5>);

    // S4: Has properties, but the initialize() declaration takes only one
    // argument.
    static_assert(!edm::HasTrivialCopyTraits<S4>);
    static_assert(edm::HasTrivialCopyProperties<S4>);
    static_assert(!edm::HasValidInitialize<S4>);
  }
}
