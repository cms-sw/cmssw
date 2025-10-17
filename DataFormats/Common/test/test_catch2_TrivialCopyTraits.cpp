#include <catch.hpp>

#include "DataFormats/Common/interface/TrivialCopyTraits.h"

#include <cmath>
#include <map>
#include <type_traits>
#include <vector>
#include <numeric>
#include <cstring>

// A trivially copyable type to test TrivialCopyTraits
struct S {
  std::string msg;
  std::vector<float> vec;
  float vec_sum = 0.0f;  // something that needs to be calculated after the copy is finished
  void setVecSum() { vec_sum = std::accumulate(vec.begin(), vec.end(), 0.0f); }
};

// Specialisation of TrivialCopyTraits for S
template <>
struct edm::TrivialCopyTraits<S> {
  using value_type = S;

  using Properties = std::array<size_t, 2>;  // {vec.size(), s.size()}

  static Properties properties(value_type const& object) {
    return std::array<size_t, 2>{{object.vec.size(), object.msg.size()}};
  }

  static void initialize(value_type& object, Properties const& sizes) {
    object.vec.resize(sizes.at(0));
    object.msg.resize(sizes.at(1));
  }

  static void finalize(value_type& object) { object.setVecSum(); }

  static std::vector<std::span<std::byte>> regions(value_type& object) {
    return {{reinterpret_cast<std::byte*>(object.msg.data()), object.msg.size()},
            {reinterpret_cast<std::byte*>(object.vec.data()), object.vec.size() * sizeof(float)}};
  }

  static std::vector<std::span<const std::byte>> regions(value_type const& object) {
    return {{reinterpret_cast<std::byte const*>(object.msg.data()), object.msg.size()},
            {reinterpret_cast<std::byte const*>(object.vec.data()), object.vec.size() * sizeof(float)}};
  }
};

// concept to check if a type is supported by TrivialCopyTraits
template <typename T>
concept HasTrivialCopyTraits = requires { typename edm::TrivialCopyTraits<T>::value_type; };

TEST_CASE("test TrivialCopyTraits", "[TrivialCopyTraits]") {
  SECTION("int") {
    REQUIRE(std::is_same<edm::TrivialCopyTraits<int>::value_type, int>::value);

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
    REQUIRE(std::is_same<edm::TrivialCopyTraits<double>::value_type, double>::value);

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
    checkDouble(sqrt(2));
    checkDouble(std::numeric_limits<double>::max());
    checkDouble(std::numeric_limits<double>::min());
    checkDouble(std::numeric_limits<double>::epsilon());
  }

  SECTION("std::vector<float>") {
    using VectorType = std::vector<float>;
    REQUIRE(std::is_same<edm::TrivialCopyTraits<VectorType>::value_type, VectorType>::value);
    REQUIRE(std::is_same<edm::TrivialCopyTraits<VectorType>::Properties, VectorType::size_type>::value);

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
    S s;
    s.vec = test_vec;
    s.msg = test_msg;
    s.setVecSum();

    REQUIRE(std::is_same<edm::TrivialCopyTraits<S>::value_type, S>::value);
    REQUIRE(std::is_same<edm::TrivialCopyTraits<S>::Properties, std::array<size_t, 2>>::value);

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
    static_assert(!HasTrivialCopyTraits<MapType>);
  }
}
