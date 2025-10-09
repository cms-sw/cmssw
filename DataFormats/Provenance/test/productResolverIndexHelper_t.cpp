#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/ProductResolverIndexHelper.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/ProductKindOfType.h"
#include "FWCore/Utilities/interface/TypeID.h"

#include <iostream>
#include <iomanip>

using namespace edm;
namespace {
  constexpr auto skipCurrentProcessLabel = edm::ProductResolverIndexHelper::skipCurrentProcessLabel();
}

TEST_CASE("ProductResolverIndexHelper", "[ProductResolverIndexHelper]") {
  TypeID typeID_ProductID(typeid(ProductID));
  TypeID typeID_EventID(typeid(EventID));

  SECTION("CreateEmpty") {
    edm::ProductResolverIndexHelper helper;
    {
      std::vector<std::string> processNames = {"processA"};
      helper.setFrozen(processNames);
    }

    REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "processA") ==
            ProductResolverIndexInvalid);
    REQUIRE(helper.index(ELEMENT_TYPE, typeID_ProductID, "labelA", "instanceA", "processA") ==
            ProductResolverIndexInvalid);
    REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "") == ProductResolverIndexInvalid);
    REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA") == ProductResolverIndexInvalid);

    edm::ProductResolverIndexHelper::Matches matches =
        helper.relatedIndexes(PRODUCT_TYPE, typeID_ProductID, "label_A", "instance_A");
    REQUIRE(matches.numberOfMatches() == 0);

    TypeID typeID(typeid(ProductID));
    REQUIRE_THROWS_AS(helper.insert(typeID, "labelA", "instanceA", "processA"), cms::Exception);
  }

  SECTION("OneEntry") {
    SECTION("Current process") {
      edm::ProductResolverIndexHelper helper;

      TypeID typeIDProductID(typeid(ProductID));
      helper.insert(typeIDProductID, "labelA", "instanceA", "processA");

      REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "processA") ==
              ProductResolverIndexInvalid);
      REQUIRE(helper.index(ELEMENT_TYPE, typeID_ProductID, "labelA", "instanceA", "processA") ==
              ProductResolverIndexInvalid);
      REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "") == ProductResolverIndexInvalid);
      REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA") == ProductResolverIndexInvalid);

      edm::ProductResolverIndexHelper::Matches matches =
          helper.relatedIndexes(PRODUCT_TYPE, typeID_ProductID, "label_A", "instance_A");
      REQUIRE(matches.numberOfMatches() == 0);

      {
        std::vector<std::string> processNames = {"processA"};
        helper.setFrozen(processNames);
      }

      matches = helper.relatedIndexes(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA");
      REQUIRE(matches.numberOfMatches() == 3);
      edm::ProductResolverIndex indexEmptyProcess = matches.index(0);
      edm::ProductResolverIndex indexSkipCurrentProcess = matches.index(1);
      edm::ProductResolverIndex indexWithProcess = matches.index(2);
      REQUIRE_THROWS_AS(matches.index(3), cms::Exception);
      REQUIRE(indexEmptyProcess < 2);
      REQUIRE(indexWithProcess < 2);
      REQUIRE(indexEmptyProcess == indexWithProcess);
      REQUIRE(indexSkipCurrentProcess == ProductResolverIndexInvalid);

      //with only one entry, all should resolve to the one with process name
      REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA") == indexWithProcess);
      REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "") == indexWithProcess);
      REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", 0) == indexWithProcess);
      REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", skipCurrentProcessLabel) ==
              indexSkipCurrentProcess);
      REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "processA") == indexWithProcess);

      REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instance", "processA") ==
              ProductResolverIndexInvalid);
      REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceAX", "processA") ==
              ProductResolverIndexInvalid);
      REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "label", "instanceA", "processA") ==
              ProductResolverIndexInvalid);
      REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelAX", "instanceA", "processA") ==
              ProductResolverIndexInvalid);
      REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "process") ==
              ProductResolverIndexInvalid);
      REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "processAX") ==
              ProductResolverIndexInvalid);
      REQUIRE(helper.index(PRODUCT_TYPE, typeID_EventID, "labelA", "instanceA", "processA") ==
              ProductResolverIndexInvalid);

      REQUIRE(helper.index(ELEMENT_TYPE, typeID_ProductID, "labelA", "instanceA") == ProductResolverIndexInvalid);
      REQUIRE(helper.index(ELEMENT_TYPE, typeID_ProductID, "labelA", "instanceA", "processA") ==
              ProductResolverIndexInvalid);

      matches = helper.relatedIndexes(ELEMENT_TYPE, typeID_ProductID, "labelA", "instanceA");
      REQUIRE(matches.numberOfMatches() == 0);

      {
        auto indexToModules = helper.indiciesForModulesInProcess("processA");
        REQUIRE(indexToModules.size() == 1);
        REQUIRE(indexToModules.count("labelA") == 1);
        auto const& range = indexToModules.equal_range("labelA");
        REQUIRE(std::get<2>(range.first->second) == indexWithProcess);
      }

      {
        auto indexToModules = helper.indiciesForModulesInProcess("processNotHere");
        REQUIRE(indexToModules.size() == 0);
      }
    }
    SECTION("Previous process") {
      edm::ProductResolverIndexHelper helper;

      TypeID typeIDProductID(typeid(ProductID));
      helper.insert(typeIDProductID, "labelA", "instanceA", "processA");

      REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "processA") ==
              ProductResolverIndexInvalid);
      REQUIRE(helper.index(ELEMENT_TYPE, typeID_ProductID, "labelA", "instanceA", "processA") ==
              ProductResolverIndexInvalid);
      REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "") == ProductResolverIndexInvalid);
      REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA") == ProductResolverIndexInvalid);

      edm::ProductResolverIndexHelper::Matches matches =
          helper.relatedIndexes(PRODUCT_TYPE, typeID_ProductID, "label_A", "instance_A");
      REQUIRE(matches.numberOfMatches() == 0);

      {
        std::vector<std::string> processNames = {"processB", "processA"};
        helper.setFrozen(processNames);
      }

      matches = helper.relatedIndexes(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA");
      REQUIRE(matches.numberOfMatches() == 3);
      edm::ProductResolverIndex indexEmptyProcess = matches.index(0);
      edm::ProductResolverIndex indexSkipCurrentProcess = matches.index(1);
      edm::ProductResolverIndex indexWithProcess = matches.index(2);
      REQUIRE_THROWS_AS(matches.index(3), cms::Exception);
      REQUIRE(indexEmptyProcess < 2);
      REQUIRE(indexWithProcess < 2);
      REQUIRE(indexEmptyProcess == indexWithProcess);
      REQUIRE(indexSkipCurrentProcess == indexWithProcess);

      //with only one entry, all should resolve to the one with process name
      REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA") == indexWithProcess);
      REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "") == indexWithProcess);
      REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", 0) == indexWithProcess);
      REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", skipCurrentProcessLabel) ==
              indexSkipCurrentProcess);
      REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "processA") == indexWithProcess);

      REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instance", "processA") ==
              ProductResolverIndexInvalid);
      REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceAX", "processA") ==
              ProductResolverIndexInvalid);
      REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "label", "instanceA", "processA") ==
              ProductResolverIndexInvalid);
      REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelAX", "instanceA", "processA") ==
              ProductResolverIndexInvalid);
      REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "process") ==
              ProductResolverIndexInvalid);
      REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "processAX") ==
              ProductResolverIndexInvalid);
      REQUIRE(helper.index(PRODUCT_TYPE, typeID_EventID, "labelA", "instanceA", "processA") ==
              ProductResolverIndexInvalid);

      REQUIRE(helper.index(ELEMENT_TYPE, typeID_ProductID, "labelA", "instanceA") == ProductResolverIndexInvalid);
      REQUIRE(helper.index(ELEMENT_TYPE, typeID_ProductID, "labelA", "instanceA", "processA") ==
              ProductResolverIndexInvalid);

      matches = helper.relatedIndexes(ELEMENT_TYPE, typeID_ProductID, "labelA", "instanceA");
      REQUIRE(matches.numberOfMatches() == 0);

      {
        auto indexToModules = helper.indiciesForModulesInProcess("processA");
        REQUIRE(indexToModules.size() == 1);
        REQUIRE(indexToModules.count("labelA") == 1);
        auto const& range = indexToModules.equal_range("labelA");
        REQUIRE(std::get<2>(range.first->second) == indexWithProcess);
      }

      {
        auto indexToModules = helper.indiciesForModulesInProcess("processNotHere");
        REQUIRE(indexToModules.size() == 0);
      }
    }
    SECTION("current and previous section") {
      edm::ProductResolverIndexHelper helper;

      TypeID typeIDProductID(typeid(ProductID));
      helper.insert(typeIDProductID, "labelA", "instanceA", "processA");
      helper.insert(typeIDProductID, "labelA", "instanceA", "processB");
      {
        std::vector<std::string> processNames = {"processB", "processA"};
        helper.setFrozen(processNames);
      }
      auto matches = helper.relatedIndexes(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA");
      REQUIRE(matches.numberOfMatches() == 4);
      edm::ProductResolverIndex indexEmptyProcess = matches.index(0);
      edm::ProductResolverIndex indexSkipCurrentProcess = matches.index(1);
      edm::ProductResolverIndex indexWithProcessA = matches.index(2);
      edm::ProductResolverIndex indexWithProcessB = matches.index(3);
      REQUIRE_THROWS_AS(matches.index(4), cms::Exception);
      REQUIRE(indexEmptyProcess == indexWithProcessB);
      REQUIRE(indexSkipCurrentProcess == indexWithProcessA);

      //with only one entry, all should resolve to the one with process name
      REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA") == indexWithProcessB);
      REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "") == indexWithProcessB);
      REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", 0) == indexWithProcessB);
      REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", skipCurrentProcessLabel) ==
              indexSkipCurrentProcess);
      REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "processA") == indexWithProcessA);
      REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "processB") == indexWithProcessB);
    }
  }

  SECTION("ManyEntries") {
    SECTION("1 simple type not in current process") {
      edm::ProductResolverIndexHelper helper;
      TypeID typeIDEventID(typeid(EventID));
      helper.insert(typeIDEventID, "labelB", "instanceB", "processB");   // 0,
      helper.insert(typeIDEventID, "label", "instanceB", "processB");    // 1,
      helper.insert(typeIDEventID, "labelX", "instanceB", "processB");   // 2,
      helper.insert(typeIDEventID, "labelB", "instance", "processB");    // 3,
      helper.insert(typeIDEventID, "labelB", "instanceX", "processB");   // 4,
      helper.insert(typeIDEventID, "labelB", "instanceB", "processB1");  // 5,
      helper.insert(typeIDEventID, "labelB", "instanceB", "processB3");  // 6,
      helper.insert(typeIDEventID, "labelB", "instanceB", "processB2");  // 7,
      helper.insert(typeIDEventID, "label", "instance", "process");      // 8,
      {
        std::vector<std::string> processNames = {
            "processC", "process", "processB", "processB1", "processB2", "processB3", "processA"};
        helper.setFrozen(processNames);
      }
      auto matches = helper.relatedIndexes(PRODUCT_TYPE, typeID_EventID, "labelB", "instanceB");
      REQUIRE(matches.numberOfMatches() == 4 + 2);  //4 process matches + 1 empty and 1 skip current process
      ProductResolverIndex indexEmptyProcess = matches.index(0);
      ProductResolverIndex indexSkipCurrentProcess = matches.index(1);
      ProductResolverIndex indexB = matches.index(2);
      ProductResolverIndex indexB1 = matches.index(3);
      ProductResolverIndex indexB2 = matches.index(4);
      ProductResolverIndex indexB3 = matches.index(5);
      REQUIRE_THROWS_AS(matches.index(6), cms::Exception);
      REQUIRE(indexB == 0);
      REQUIRE(indexB1 == 5);
      REQUIRE(indexB2 == 7);
      REQUIRE(indexB3 == 6);
      REQUIRE(indexEmptyProcess == indexB);
      REQUIRE(indexSkipCurrentProcess == indexB);

      REQUIRE(std::string(matches.moduleLabel(5)) == "labelB");
      REQUIRE(std::string(matches.productInstanceName(5)) == "instanceB");
      REQUIRE(std::string(matches.processName(5)) == "processB3");
      REQUIRE(std::string(matches.processName(0)) == "");
    }
    SECTION("1 simple type in current process") {
      edm::ProductResolverIndexHelper helper;
      TypeID typeIDEventID(typeid(EventID));
      helper.insert(typeIDEventID, "labelB", "instanceB", "processB");   // 0,
      helper.insert(typeIDEventID, "label", "instanceB", "processB");    // 1,
      helper.insert(typeIDEventID, "labelX", "instanceB", "processB");   // 2,
      helper.insert(typeIDEventID, "labelB", "instance", "processB");    // 3,
      helper.insert(typeIDEventID, "labelB", "instanceX", "processB");   // 4,
      helper.insert(typeIDEventID, "labelB", "instanceB", "processB1");  // 5,
      helper.insert(typeIDEventID, "labelB", "instanceB", "processB3");  // 6,
      helper.insert(typeIDEventID, "labelB", "instanceB", "processB2");  // 7,
      {
        std::vector<std::string> processNames = {"processB", "processB1", "processB2", "processB3", "processA"};
        helper.setFrozen(processNames);
      }
      auto matches = helper.relatedIndexes(PRODUCT_TYPE, typeID_EventID, "labelB", "instanceB");
      REQUIRE(matches.numberOfMatches() == 4 + 2);  //4 process matches + 1 empty and 1 skip current process
      ProductResolverIndex indexEmptyProcess = matches.index(0);
      ProductResolverIndex indexSkipCurrentProcess = matches.index(1);
      ProductResolverIndex indexB = matches.index(2);
      ProductResolverIndex indexB1 = matches.index(3);
      ProductResolverIndex indexB2 = matches.index(4);
      ProductResolverIndex indexB3 = matches.index(5);
      REQUIRE_THROWS_AS(matches.index(6), cms::Exception);
      REQUIRE(indexB == 0);
      REQUIRE(indexB1 == 5);
      REQUIRE(indexB2 == 7);
      REQUIRE(indexB3 == 6);
      REQUIRE(indexEmptyProcess == indexB);
      REQUIRE(indexSkipCurrentProcess == indexB1);

      REQUIRE(std::string(matches.moduleLabel(5)) == "labelB");
      REQUIRE(std::string(matches.productInstanceName(5)) == "instanceB");
      REQUIRE(std::string(matches.processName(5)) == "processB3");
      REQUIRE(std::string(matches.processName(0)) == "");
    }

    SECTION("2 simple types") {
      edm::ProductResolverIndexHelper helper;
      TypeID typeIDProductID(typeid(ProductID));
      TypeID typeIDEventID(typeid(EventID));
      helper.insert(typeIDEventID, "labelB", "instanceB", "processB");    // 0,
      helper.insert(typeIDEventID, "label", "instanceB", "processB");     // 1,
      helper.insert(typeIDEventID, "labelX", "instanceB", "processB");    // 2,
      helper.insert(typeIDEventID, "labelB", "instance", "processB");     // 3,
      helper.insert(typeIDEventID, "labelB", "instanceX", "processB");    // 4,
      helper.insert(typeIDEventID, "labelB", "instanceB", "processB1");   // 5,
      helper.insert(typeIDEventID, "labelB", "instanceB", "processB3");   // 6,
      helper.insert(typeIDEventID, "labelB", "instanceB", "processB2");   // 7,
      helper.insert(typeIDProductID, "label", "instance", "process");     // 8,
      helper.insert(typeIDEventID, "label", "instance", "process");       // 9,
      helper.insert(typeIDProductID, "labelA", "instanceA", "processA");  // 10
      {
        std::vector<std::string> processNames = {
            "processC", "process", "processB", "processB1", "processB2", "processB3", "processA"};
        helper.setFrozen(processNames);
      }
      auto matches = helper.relatedIndexes(PRODUCT_TYPE, typeID_EventID, "labelB", "instanceB");
      REQUIRE(matches.numberOfMatches() == 4 + 2);
      ProductResolverIndex indexEmptyProcess = matches.index(0);
      ProductResolverIndex indexSkipCurrentProcess = matches.index(1);
      ProductResolverIndex indexB = matches.index(2);
      ProductResolverIndex indexB1 = matches.index(3);
      ProductResolverIndex indexB2 = matches.index(4);
      ProductResolverIndex indexB3 = matches.index(5);
      REQUIRE_THROWS_AS(matches.index(6), cms::Exception);
      REQUIRE(indexB == 0);
      REQUIRE(indexB1 == 5);
      REQUIRE(indexB2 == 7);
      REQUIRE(indexB3 == 6);
      REQUIRE(indexEmptyProcess == indexB);
      REQUIRE(indexSkipCurrentProcess == indexB);

      REQUIRE(std::string(matches.moduleLabel(5)) == "labelB");
      REQUIRE(std::string(matches.productInstanceName(5)) == "instanceB");
      REQUIRE(std::string(matches.processName(5)) == "processB3");
      REQUIRE(std::string(matches.processName(0)) == "");
    }
    SECTION("many types and entries") {
      edm::ProductResolverIndexHelper helper;

      TypeID typeIDProductID(typeid(ProductID));
      TypeID typeIDEventID(typeid(EventID));
      TypeID typeIDVectorInt(typeid(std::vector<int>));
      TypeID typeIDSetInt(typeid(std::set<int>));
      TypeID typeIDVSimpleDerived(typeid(std::vector<edmtest::SimpleDerived>));
      // order of indicies: <full process> [ same but for each element type]
      helper.insert(typeIDVectorInt, "labelC", "instanceC", "processC");  // 0,
      helper.insert(typeIDVectorInt, "label", "instance", "process");     // 1,
      helper.insert(typeIDEventID, "labelB", "instanceB", "processB");    // 2,
      helper.insert(typeIDEventID, "label", "instanceB", "processB");     // 3,
      helper.insert(typeIDEventID, "labelX", "instanceB", "processB");    // 4,
      helper.insert(typeIDEventID, "labelB", "instance", "processB");     // 5,
      helper.insert(typeIDEventID, "labelB", "instanceX", "processB");    // 6,
      helper.insert(typeIDEventID, "labelB", "instanceB", "processB1");   // 7,
      helper.insert(typeIDEventID, "labelB", "instanceB", "processB3");   // 8,
      helper.insert(typeIDEventID, "labelB", "instanceB", "processB2");   // 9,
      helper.insert(typeIDProductID, "label", "instance", "process");     // 10,
      helper.insert(typeIDEventID, "label", "instance", "process");       // 11,
      helper.insert(typeIDProductID, "labelA", "instanceA", "processA");  // 12
      REQUIRE_THROWS_AS(helper.insert(typeIDProductID, "labelA", "instanceA", "processA"),
                        cms::Exception);  // duplicate

      helper.insert(typeIDSetInt, "labelC", "instanceC", "processC");  // 13

      helper.insert(typeIDVSimpleDerived, "labelC", "instanceC", "processC");  // 14

      {
        std::vector<std::string> processNames = {
            "processC", "process", "processB", "processB1", "processB2", "processB3", "processA"};
        helper.setFrozen(processNames);
      }

      TypeID typeID_int(typeid(int));
      REQUIRE(helper.index(ELEMENT_TYPE, typeID_int, "labelC", "instanceC", "processC") ==
              ProductResolverIndexAmbiguous);
      REQUIRE(helper.index(ELEMENT_TYPE, typeID_int, "labelC", "instanceC", "processQ") == ProductResolverIndexInvalid);
      REQUIRE(helper.index(ELEMENT_TYPE, typeID_int, "labelC", "instanceC") == ProductResolverIndexAmbiguous);

      TypeID typeID_vint(typeid(std::vector<int>));
      REQUIRE(helper.index(PRODUCT_TYPE, typeID_vint, "labelC", "instanceC", "processC") == 0);
      REQUIRE(helper.index(PRODUCT_TYPE, typeID_vint, "labelC", "instanceC") == 0);  //only one with no process

      TypeID typeID_sint(typeid(std::set<int>));
      REQUIRE(helper.index(PRODUCT_TYPE, typeID_sint, "labelC", "instanceC", "processC") == 13);
      REQUIRE(helper.index(PRODUCT_TYPE, typeID_sint, "labelC", "instanceC") == 13);  //only one with no process

      TypeID typeID_Simple(typeid(edmtest::Simple));
      REQUIRE(helper.index(ELEMENT_TYPE, typeID_Simple, "labelC", "instanceC") == 14);  //only one with no process
      REQUIRE(helper.index(ELEMENT_TYPE, typeID_Simple, "labelC", "instanceC", "processC") == 14);

      TypeID typeID_SimpleDerived(typeid(edmtest::SimpleDerived));
      REQUIRE(helper.index(ELEMENT_TYPE, typeID_SimpleDerived, "labelC", "instanceC") ==
              14);  //only one with no process
      REQUIRE(helper.index(ELEMENT_TYPE, typeID_SimpleDerived, "labelC", "instanceC", "processC") == 14);

      TypeID typeID_VSimpleDerived(typeid(std::vector<edmtest::SimpleDerived>));
      REQUIRE(helper.index(PRODUCT_TYPE, typeID_VSimpleDerived, "labelC", "instanceC") ==
              14);  //only one with no process
      REQUIRE(helper.index(PRODUCT_TYPE, typeID_VSimpleDerived, "labelC", "instanceC", "processC") == 14);

      auto matches = helper.relatedIndexes(PRODUCT_TYPE, typeID_EventID, "labelB", "instanceB");
      REQUIRE(matches.numberOfMatches() == 4 + 2);  //4 process matches + 1 empty and 1 skip current process
      ProductResolverIndex indexEmptyProcess = matches.index(0);
      ProductResolverIndex indexSkipCurrentProcess = matches.index(1);
      ProductResolverIndex indexB = matches.index(2);
      ProductResolverIndex indexB1 = matches.index(3);
      ProductResolverIndex indexB2 = matches.index(4);
      ProductResolverIndex indexB3 = matches.index(5);
      REQUIRE_THROWS_AS(matches.index(6), cms::Exception);
      REQUIRE(indexEmptyProcess == 2);
      REQUIRE(indexSkipCurrentProcess == 2);
      REQUIRE(indexB == 2);
      REQUIRE(indexB1 == 7);
      REQUIRE(indexB2 == 9);
      REQUIRE(indexB3 == 8);

      REQUIRE(std::string(matches.moduleLabel(5)) == "labelB");
      REQUIRE(std::string(matches.productInstanceName(5)) == "instanceB");
      REQUIRE(std::string(matches.processName(5)) == "processB3");
      REQUIRE(std::string(matches.processName(0)) == "");

      {
        auto indexToModules = helper.indiciesForModulesInProcess("processA");
        REQUIRE(indexToModules.size() == 1);
      }
      {
        auto indexToModules = helper.indiciesForModulesInProcess("processB");
        REQUIRE(indexToModules.size() == 5);
      }
      {
        auto indexToModules = helper.indiciesForModulesInProcess("processB1");
        REQUIRE(indexToModules.size() == 1);
      }
      {
        auto indexToModules = helper.indiciesForModulesInProcess("processB2");
        REQUIRE(indexToModules.size() == 1);
      }
      {
        auto indexToModules = helper.indiciesForModulesInProcess("processB3");
        REQUIRE(indexToModules.size() == 1);
      }
      {
        auto indexToModules = helper.indiciesForModulesInProcess("processC");
        REQUIRE(indexToModules.size() == 3);
      }
    }
  }
}
