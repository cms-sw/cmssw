/*----------------------------------------------------------------------

Test program for edm::SoATuple class.
Changed by Viji on 29-06-2005

 ----------------------------------------------------------------------*/

#include <catch2/catch_all.hpp>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include "FWCore/Utilities/interface/SoATuple.h"

namespace {
  struct ConstructDestructCounter {
    static unsigned int s_constructorCalls;
    static unsigned int s_destructorCalls;

    //make sure the destructor is being called on the correct memory location
    void* originalThis;

    ConstructDestructCounter() : originalThis(this) { ++s_constructorCalls; }
    ConstructDestructCounter(ConstructDestructCounter const&) : originalThis(this) { ++s_constructorCalls; }
    ConstructDestructCounter(ConstructDestructCounter&&) : originalThis(this) { ++s_constructorCalls; }
    ~ConstructDestructCounter() {
      REQUIRE(originalThis == this);
      ++s_destructorCalls;
    }
  };
  unsigned int ConstructDestructCounter::s_constructorCalls = 0;
  unsigned int ConstructDestructCounter::s_destructorCalls = 0;

  struct CharDummy {
    char i;
    char j;
  };

  struct ComplexDummy {
    char* p;
    double f;
  };
}  // namespace

TEST_CASE("edm::SoATuple", "[SoATuple]") {
  SECTION("builtinTest") {
    edm::SoATuple<int, float, bool> s;
    s.reserve(3);
    REQUIRE(s.size() == 0);

    s.push_back(std::make_tuple(int{1}, float{3.2}, false));
    REQUIRE(s.size() == 1);
    REQUIRE(1 == s.get<0>(0));
    REQUIRE(float{3.2} == s.get<1>(0));
    REQUIRE(false == s.get<2>(0));

    s.push_back(std::make_tuple(int{2}, float{3.1415}, true));
    REQUIRE(s.size() == 2);
    REQUIRE(1 == s.get<0>(0));
    REQUIRE(float{3.2} == s.get<1>(0));
    REQUIRE(false == s.get<2>(0));
    REQUIRE(2 == s.get<0>(1));
    REQUIRE(float{3.1415} == s.get<1>(1));
    REQUIRE(true == s.get<2>(1));

    s.push_back(std::make_tuple(int{-1}, float{58.6}, true));
    REQUIRE(s.size() == 3);
    REQUIRE(1 == s.get<0>(0));
    REQUIRE(float{3.2} == s.get<1>(0));
    REQUIRE(false == s.get<2>(0));
    REQUIRE(2 == s.get<0>(1));
    REQUIRE(float{3.1415} == s.get<1>(1));
    REQUIRE(true == s.get<2>(1));
    REQUIRE(-1 == s.get<0>(2));
    REQUIRE(float{58.6} == s.get<1>(2));
    REQUIRE(true == s.get<2>(2));
  }

  SECTION("badPaddingTest") {
    edm::SoATuple<bool, int, double> s;
    s.reserve(3);
    REQUIRE(s.size() == 0);

    s.push_back(std::make_tuple(false, int{1}, double{3.2}));
    REQUIRE(s.size() == 1);
    REQUIRE(1 == s.get<1>(0));
    REQUIRE(double{3.2} == s.get<2>(0));
    REQUIRE(false == s.get<0>(0));

    s.push_back(std::make_tuple(true, int{2}, double{3.1415}));
    REQUIRE(s.size() == 2);
    REQUIRE(1 == s.get<1>(0));
    REQUIRE(double{3.2} == s.get<2>(0));
    REQUIRE(false == s.get<0>(0));
    REQUIRE(2 == s.get<1>(1));
    REQUIRE(double{3.1415} == s.get<2>(1));
    REQUIRE(true == s.get<0>(1));

    s.push_back(std::make_tuple(true, int{-1}, double{58.6}));
    REQUIRE(s.size() == 3);
    REQUIRE(1 == s.get<1>(0));
    REQUIRE(double{3.2} == s.get<2>(0));
    REQUIRE(false == s.get<0>(0));
    REQUIRE(2 == s.get<1>(1));
    REQUIRE(double{3.1415} == s.get<2>(1));
    REQUIRE(true == s.get<0>(1));
    REQUIRE(-1 == s.get<1>(2));
    REQUIRE(double{58.6} == s.get<2>(2));
    REQUIRE(true == s.get<0>(2));
  }

  SECTION("classTest") {
    ConstructDestructCounter::s_constructorCalls = 0;
    ConstructDestructCounter::s_destructorCalls = 0;
    REQUIRE(ConstructDestructCounter::s_constructorCalls == ConstructDestructCounter::s_destructorCalls);
    {
      edm::SoATuple<std::string, ConstructDestructCounter> s;
      s.reserve(3);
      const std::string kFoo{"foo"};
      REQUIRE(s.size() == 0);
      {
        ConstructDestructCounter dummy;
        s.push_back(std::make_tuple(kFoo, dummy));
      }
      REQUIRE(ConstructDestructCounter::s_constructorCalls == ConstructDestructCounter::s_destructorCalls + 1);
      REQUIRE(s.size() == 1);
      REQUIRE(kFoo == s.get<0>(0));

      const std::string kBar{"bar"};
      {
        ConstructDestructCounter dummy;
        s.push_back(std::make_tuple(kBar, dummy));
      }
      REQUIRE(ConstructDestructCounter::s_constructorCalls == ConstructDestructCounter::s_destructorCalls + 2);
      REQUIRE(s.size() == 2);
      REQUIRE(kFoo == s.get<0>(0));
      REQUIRE(kBar == s.get<0>(1));
    }
    REQUIRE(ConstructDestructCounter::s_constructorCalls == ConstructDestructCounter::s_destructorCalls);
  }

  SECTION("growthTest") {
    ConstructDestructCounter::s_constructorCalls = 0;
    ConstructDestructCounter::s_destructorCalls = 0;
    REQUIRE(ConstructDestructCounter::s_constructorCalls == ConstructDestructCounter::s_destructorCalls);
    {
      edm::SoATuple<unsigned int, ConstructDestructCounter> s;
      REQUIRE(s.size() == 0);
      REQUIRE(s.capacity() == 0);
      for (unsigned int i = 0; i < 100; ++i) {
        {
          ConstructDestructCounter dummy;
          s.push_back(std::make_tuple(i, dummy));
        }
        REQUIRE(ConstructDestructCounter::s_constructorCalls == ConstructDestructCounter::s_destructorCalls + i + 1);
        REQUIRE(s.size() == i + 1);
        REQUIRE(s.capacity() >= i + 1);
        for (unsigned int j = 0; j < s.size(); ++j) {
          REQUIRE(j == s.get<0>(j));
        }
      }

      s.shrink_to_fit();
      REQUIRE(s.capacity() == s.size());
      REQUIRE(ConstructDestructCounter::s_constructorCalls - ConstructDestructCounter::s_destructorCalls == s.size());
      for (unsigned int j = 0; j < s.size(); ++j) {
        REQUIRE(j == s.get<0>(j));
      }
    }

    REQUIRE(ConstructDestructCounter::s_constructorCalls == ConstructDestructCounter::s_destructorCalls);
  }

  SECTION("copyConstructorTest") {
    ConstructDestructCounter::s_constructorCalls = 0;
    ConstructDestructCounter::s_destructorCalls = 0;
    REQUIRE(ConstructDestructCounter::s_constructorCalls == ConstructDestructCounter::s_destructorCalls);
    {
      edm::SoATuple<unsigned int, ConstructDestructCounter> s;
      REQUIRE(s.size() == 0);
      REQUIRE(s.capacity() == 0);
      for (unsigned int i = 0; i < 100; ++i) {
        {
          ConstructDestructCounter dummy;
          s.push_back(std::make_tuple(i, dummy));
        }
        edm::SoATuple<unsigned int, ConstructDestructCounter> sCopy(s);
        REQUIRE(ConstructDestructCounter::s_constructorCalls ==
                ConstructDestructCounter::s_destructorCalls + 2 * i + 2);
        REQUIRE(s.size() == i + 1);
        REQUIRE(s.capacity() >= i + 1);
        REQUIRE(sCopy.size() == i + 1);
        REQUIRE(sCopy.capacity() >= i + 1);

        for (unsigned int j = 0; j < s.size(); ++j) {
          REQUIRE(j == s.get<0>(j));
          REQUIRE(j == sCopy.get<0>(j));
        }
      }
    }
    REQUIRE(ConstructDestructCounter::s_constructorCalls == ConstructDestructCounter::s_destructorCalls);
  }

  SECTION("assignmentTest") {
    const std::vector<std::string> sValues = {"foo", "fii", "fee"};
    edm::SoATuple<std::string, int> s;
    s.reserve(sValues.size());
    int i = 0;
    for (auto const& v : sValues) {
      s.push_back(std::make_tuple(v, i));
      ++i;
    }

    edm::SoATuple<std::string, int> sAssign;
    sAssign.reserve(2);
    sAssign.push_back(std::make_tuple("barney", 10));
    sAssign.push_back(std::make_tuple("fred", 7));
    REQUIRE(sAssign.size() == 2);

    sAssign = s;
    REQUIRE(sAssign.size() == s.size());
    REQUIRE(sAssign.size() == sValues.size());

    i = 0;
    for (auto const& v : sValues) {
      REQUIRE(v == sAssign.get<0>(i));
      REQUIRE(i == sAssign.get<1>(i));
      ++i;
    }
  }

  SECTION("moveAssignmentTest") {
    const std::vector<std::string> sValues = {"foo", "fii", "fee"};
    edm::SoATuple<std::string, int> s;
    s.reserve(sValues.size());
    int i = 0;
    for (auto const& v : sValues) {
      s.push_back(std::make_tuple(v, i));
      ++i;
    }

    edm::SoATuple<std::string, int> sAssign;
    sAssign.reserve(2);
    sAssign.push_back(std::make_tuple("barney", 10));
    sAssign.push_back(std::make_tuple("fred", 7));
    REQUIRE(sAssign.size() == 2);

    sAssign = std::move(s);
    REQUIRE(0 == s.size());
    REQUIRE(sAssign.size() == sValues.size());

    i = 0;
    for (auto const& v : sValues) {
      REQUIRE(v == sAssign.get<0>(i));
      REQUIRE(i == sAssign.get<1>(i));
      ++i;
    }
  }

  SECTION("loopTest") {
    edm::SoATuple<int, int, int> s;
    s.reserve(50);
    for (int i = 0; i < 50; ++i) {
      s.push_back(std::make_tuple(i, i + 1, i + 2));
    }
    REQUIRE(50 == s.size());
    int index = 0;
    for (auto it = s.begin<0>(), itEnd = s.end<0>(); it != itEnd; ++it, ++index) {
      REQUIRE(index == *it);
    }
    index = 1;
    for (auto it = s.begin<1>(), itEnd = s.end<1>(); it != itEnd; ++it, ++index) {
      REQUIRE(index == *it);
    }

    index = 2;
    for (auto it = s.begin<2>(), itEnd = s.end<2>(); it != itEnd; ++it, ++index) {
      REQUIRE(index == *it);
    }
  }

  SECTION("emplace_backTest") {
    const std::vector<std::string> sValues = {"foo", "fii", "fee"};
    edm::SoATuple<std::string, int> s;
    s.reserve(sValues.size());
    int i = 0;
    for (auto const& v : sValues) {
      s.emplace_back(v, i);
      ++i;
    }
    i = 0;
    for (auto const& v : sValues) {
      REQUIRE(v == s.get<0>(i));
      REQUIRE(i == s.get<1>(i));
      ++i;
    }
  }

  SECTION("alignmentTest") {
    REQUIRE((alignof(double) == edm::soahelper::SoATupleHelper<double, bool>::max_alignment));
    REQUIRE((alignof(double) == edm::soahelper::SoATupleHelper<bool, double>::max_alignment));
    REQUIRE((alignof(float) == edm::soahelper::SoATupleHelper<float, bool>::max_alignment));
    REQUIRE((alignof(float) == edm::soahelper::SoATupleHelper<bool, float>::max_alignment));
    REQUIRE((alignof(CharDummy) == edm::soahelper::SoATupleHelper<char, CharDummy>::max_alignment));
    REQUIRE((alignof(CharDummy) == edm::soahelper::SoATupleHelper<CharDummy, char>::max_alignment));
    REQUIRE((alignof(ComplexDummy) == edm::soahelper::SoATupleHelper<char, ComplexDummy>::max_alignment));
    REQUIRE((alignof(ComplexDummy) == edm::soahelper::SoATupleHelper<ComplexDummy, char>::max_alignment));

    REQUIRE((alignof(float) == edm::soahelper::SoATupleHelper<float, float>::max_alignment));
    REQUIRE((16 == edm::soahelper::SoATupleHelper<edm::AlignedVec<float>, edm::AlignedVec<float>>::max_alignment));

    REQUIRE((alignof(double) == edm::soahelper::SoATupleHelper<double, double>::max_alignment));
    REQUIRE((16 == edm::soahelper::SoATupleHelper<edm::AlignedVec<double>, edm::AlignedVec<double>>::max_alignment));

    edm::SoATuple<edm::AlignedVec<float>, edm::AlignedVec<float>, edm::AlignedVec<float>> vFloats;
    vFloats.reserve(50);
    for (unsigned int i = 0; i < 50; ++i) {
      vFloats.emplace_back(1.0f, 2.0f, 3.0f);
    }
    REQUIRE(reinterpret_cast<std::intptr_t>(vFloats.begin<0>()) % 16 == 0);
    REQUIRE(reinterpret_cast<std::intptr_t>(vFloats.begin<1>()) % 16 == 0);
    REQUIRE(reinterpret_cast<std::intptr_t>(vFloats.begin<2>()) % 16 == 0);
  }
}
