#include <cassert>
#include <iostream>
#include <memory>
#include <vector>

#include <catch.hpp>

#include "DataFormats/Common/interface/Wrapper.h"

class CopyNoMove {
public:
  CopyNoMove() = default;
  CopyNoMove(CopyNoMove const&) { /* std::cout << "copied\n"; */ }
  CopyNoMove& operator=(CopyNoMove const&) { /*std::cout << "assigned\n";*/ return *this; }
};

class MoveNoCopy {
public:
  MoveNoCopy() = default;
  MoveNoCopy(MoveNoCopy const&) = delete;
  MoveNoCopy& operator=(MoveNoCopy const&) = delete;
  MoveNoCopy(MoveNoCopy&&) { /* std::cout << "moved\n";*/ }
  MoveNoCopy& operator=(MoveNoCopy&&) { /* std::cout << "moved\n";*/ return *this; }
};

class NoDefaultCtor {
public:
  NoDefaultCtor() = delete;
  NoDefaultCtor(edm::Uninitialized) {}
};

TEST_CASE("test Wrapper", "[Wrapper]") {
  auto thing = std::make_unique<CopyNoMove>();
  edm::Wrapper<CopyNoMove> wrap(std::move(thing));

  auto thing2 = std::make_unique<MoveNoCopy>();
  edm::Wrapper<MoveNoCopy> wrap2(std::move(thing2));

  auto thing3 = std::make_unique<std::vector<double>>(10, 2.2);
  REQUIRE(thing3->size() == 10);

  edm::Wrapper<std::vector<double>> wrap3(std::move(thing3));
  REQUIRE(wrap3->size() == 10);
  REQUIRE(thing3.get() == 0);

  auto thing4 = std::make_unique<NoDefaultCtor>(edm::kUninitialized);
  edm::Wrapper<NoDefaultCtor> wrap4(std::move(thing4));
}
