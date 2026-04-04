/*
 *  sharedresourcesregistry_t.catch2.cc
 *  CMSSW
 *
 *  Created by Chris Jones on 8/8/05.
 *
 */

#include "catch2/catch_all.hpp"

#define SHAREDRESOURCETESTACCESSORS 1
#include "FWCore/Framework/interface/SharedResourcesRegistry.h"
#include "FWCore/Framework/interface/SharedResourcesAcquirer.h"

using namespace edm;

class testSharedResourcesRegistry {
public:
  edm::SharedResourcesRegistry registry;
};

TEST_CASE("SharedResourcesRegistry", "[Framework]") {
  SECTION("oneTest") {
    testSharedResourcesRegistry test;
    edm::SharedResourcesRegistry& reg = test.registry;

    REQUIRE(reg.resourceMap().size() == 0);

    reg.registerSharedResource("foo");
    reg.registerSharedResource("bar");
    reg.registerSharedResource("zoo");

    {
      std::vector<std::string> res{"foo", "bar", "zoo"};
      auto tester = reg.createAcquirer(res);

      REQUIRE(1 == tester.numberOfResources());
    }
    {
      std::vector<std::string> res{"foo"};
      auto tester = reg.createAcquirer(res);

      REQUIRE(1 == tester.numberOfResources());
    }
  }

  SECTION("multipleTest") {
    testSharedResourcesRegistry test;
    edm::SharedResourcesRegistry& reg = test.registry;
    auto const& resourceMap = reg.resourceMap();

    reg.registerSharedResource("foo");
    reg.registerSharedResource("bar");
    reg.registerSharedResource("zoo");
    reg.registerSharedResource("zoo");
    reg.registerSharedResource("bar");
    reg.registerSharedResource("zoo");

    REQUIRE(resourceMap.at(std::string("foo")).first.get() == nullptr);
    REQUIRE(resourceMap.at(std::string("zoo")).first.get() != nullptr);
    REQUIRE(resourceMap.at(std::string("bar")).first.get() != nullptr);
    REQUIRE(resourceMap.at(std::string("bar")).second == 2);
    REQUIRE(resourceMap.at(std::string("foo")).second == 1);
    REQUIRE(resourceMap.at(std::string("zoo")).second == 3);
    REQUIRE(resourceMap.size() == 3);

    {
      std::vector<std::string> res{"foo", "bar", "zoo"};
      auto tester = reg.createAcquirer(res);

      REQUIRE(2 == tester.numberOfResources());
    }
    {
      std::vector<std::string> res{"foo"};
      auto tester = reg.createAcquirer(res);

      REQUIRE(1 == tester.numberOfResources());
    }
    {
      std::vector<std::string> res{"bar"};
      auto tester = reg.createAcquirer(res);

      REQUIRE(1 == tester.numberOfResources());
    }
  }
}
