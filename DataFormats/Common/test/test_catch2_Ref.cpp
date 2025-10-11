#include "SimpleEDProductGetter.h"

#include "DataFormats/Common/interface/EDProductGetter.h"
#include "DataFormats/Common/interface/Ref.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "catch2/catch_all.hpp"

#include <iostream>
#include <string>
#include <utility>
#include <thread>
#include <atomic>

using product1_t = std::vector<int>;
using product2_t = std::map<std::string, int>;
using ref1_t = edm::Ref<product1_t>;
// using ref2_t = edm::Ref<product2_t, int>;

TEST_CASE("Ref tests", "[Ref]") {
  SECTION("default constructor") {
    ref1_t default_ref;
    REQUIRE(default_ref.isNull());
    REQUIRE(default_ref.isNonnull() == false);
    REQUIRE(!default_ref);
    REQUIRE(default_ref.productGetter() == 0);
    REQUIRE(default_ref.id().isValid() == false);
    REQUIRE(default_ref.isAvailable() == false);
  }

  // SECTION("default constructor with string key") {
  //   ref2_t default_ref;
  //   REQUIRE(default_ref.isNull());
  //   REQUIRE(default_ref.isNonnull()==false);
  //   REQUIRE(!default_ref);
  //   REQUIRE(default_ref.productGetter()==0);
  //   REQUIRE(default_ref.id().isValid()==false);
  //   REQUIRE(default_ref.id().isAvailable()==false);
  // }

  SECTION("non-default constructor") {
    SimpleEDProductGetter getter;
    edm::ProductID id(1, 201U);
    REQUIRE(id.isValid());
    auto prod = std::make_unique<product1_t>();
    prod->push_back(1);
    prod->push_back(2);
    getter.addProduct(id, std::move(prod));

    ref1_t ref0(id, 0, &getter);
    REQUIRE(ref0.isNull() == false);
    REQUIRE(ref0.isNonnull());
    REQUIRE(!!ref0);
    REQUIRE(ref0.productGetter() == &getter);
    REQUIRE(ref0.id().isValid());
    REQUIRE(ref0.isAvailable() == true);
    REQUIRE(*ref0 == 1);

    ref1_t ref1(id, 1, &getter);
    REQUIRE(ref1.isNonnull());
    REQUIRE(ref1.isAvailable() == true);
    REQUIRE(*ref1 == 2);
  }

  // SECTION("non-default constructor 2") {
  //   SimpleEDProductGetter getter;
  //   edm::EDProductGetter::Operate op(&getter);
  //   edm::ProductID id(1, 201U);
  //   REQUIRE(id.isValid());
  //   auto prod = std::make_unique<product2_t>();
  //   prod->insert(std::make_pair(std::string("a"), 1));
  //   prod->insert(std::make_pair(std::string("b"), 2));
  //   prod->insert(std::make_pair(std::string("c"), 3));
  //   getter.addProduct(id, prod);
  //   ref2_t refa(id, std::string("a"), &getter);
  //   REQUIRE(refa.isNull()==false);
  //   REQUIRE(refa.isNonnull());
  //   REQUIRE(!!refa);
  //   REQUIRE(refa.productGetter()==&getter);
  //   REQUIRE(refa.id().isValid());
  //   REQUIRE(*refa == 1);
  //   ref2_t refb(id, "b", &getter);
  //   REQUIRE(refb.isNonnull());
  //   REQUIRE(*refb == 2);
  // }

  SECTION("using wrong productid") {
    SimpleEDProductGetter getter;
    edm::ProductID id(1, 1U);
    REQUIRE(id.isValid());
    auto prod = std::make_unique<product1_t>();
    prod->push_back(1);
    prod->push_back(2);
    getter.addProduct(id, std::move(prod));
    edm::ProductID wrong_id(1, 100U);
    REQUIRE(wrong_id.isValid());
    ref1_t ref(wrong_id, 0, &getter);
    REQUIRE_THROWS_AS(*ref, edm::Exception);
    REQUIRE_THROWS_AS(ref.operator->(), edm::Exception);
  }

  SECTION("threading") {
    constexpr int kNThreads = 8;
    std::atomic<int> s_threadsStarting{kNThreads};
    SimpleEDProductGetter getter;
    edm::ProductID id(1, 1U);
    REQUIRE(id.isValid());
    auto prod = std::make_unique<product1_t>();
    prod->push_back(1);
    prod->push_back(2);
    getter.addProduct(id, std::move(prod));
    ref1_t ref0(id, 0, &getter);
    ref1_t ref1(id, 1, &getter);
    std::vector<std::thread> threads;
    std::vector<std::exception_ptr> excepPtrs(kNThreads, std::exception_ptr{});
    for (unsigned int i = 0; i < kNThreads; ++i) {
      threads.emplace_back([&ref0, &ref1, i, &excepPtrs, &s_threadsStarting]() {
        --s_threadsStarting;
        while (0 != s_threadsStarting) {
        }
        try {
          REQUIRE(*ref0 == 1);
          REQUIRE(*ref1 == 2);
        } catch (...) {
          excepPtrs[i] = std::current_exception();
        }
      });
    }
    for (auto& t : threads) {
      t.join();
    }
    for (auto& e : excepPtrs) {
      if (e) {
        std::rethrow_exception(e);
      }
    }
  }
}
