#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "DataFormats/Common/interface/DeviceProduct.h"

namespace {
  class Metadata {
  public:
    Metadata(int v = 0) : value_(v) {}
    void synchronize(int& ret) const { ret = value_; }

  private:
    int const value_;
  };

  class Product {
  public:
    Product() = default;
    Product(int v) : value_(v) {}

    int value() const { return value_; }

  private:
    int value_ = 0;
  };

  class ProductNoDefault {
  public:
    ProductNoDefault(int v) : value_(v) {}

    int value() const { return value_; }

  private:
    int value_;
  };

}  // namespace

TEST_CASE("DeviceProduct", "[DeviceProduct]") {
  SECTION("Default constructor") { [[maybe_unused]] edm::DeviceProduct<int> prod; }

  SECTION("Default-constructible data product") {
    SECTION("Default constructed") {
      edm::DeviceProduct<Product> prod(std::make_shared<Metadata>(3));

      int md = 0;
      auto const& data = prod.getSynchronized<Metadata>(md);
      REQUIRE(md == 3);
      REQUIRE(data.value() == 0);
    }

    SECTION("Explicitly constructed") {
      edm::DeviceProduct<Product> prod(std::make_shared<Metadata>(4), 42);

      int md = 0;
      auto const& data = prod.getSynchronized<Metadata>(md);
      REQUIRE(md == 4);
      REQUIRE(data.value() == 42);
    }
  }

  SECTION("Non-default-constructible data product") {
    edm::DeviceProduct<ProductNoDefault> prod(std::make_shared<Metadata>(5), 314);

    int md = 0;
    auto const& data = prod.getSynchronized<Metadata>(md);
    REQUIRE(md == 5);
    REQUIRE(data.value() == 314);
  }
}
