/*
 *  productResolverIndexHelper_t.cppunit.cc
 */
#define CATCH_CONFIG_MAIN
#include <catch.hpp>

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






TEST_CASE("ProductResolverIndexHelper", "[ProductResolverIndexHelper]") {
  TypeID typeID_ProductID(typeid(ProductID));
  TypeID typeID_EventID(typeid(EventID));

  SECTION("CreateEmpty") {
    edm::ProductResolverIndexHelper helper;
    helper.setFrozen("processA");

    REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "processA") == ProductResolverIndexInvalid);
    REQUIRE(helper.index(ELEMENT_TYPE, typeID_ProductID, "labelA", "instanceA", "processA") == ProductResolverIndexInvalid);
    REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "") == ProductResolverIndexInvalid);
    REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA") == ProductResolverIndexInvalid);

    edm::ProductResolverIndexHelper::Matches matches = helper.relatedIndexes(PRODUCT_TYPE, typeID_ProductID, "label_A", "instance_A");
    REQUIRE(matches.numberOfMatches() == 0);
    matches = helper.relatedIndexes(PRODUCT_TYPE, typeID_ProductID);
    REQUIRE(matches.numberOfMatches() == 0);

    TypeID typeID(typeid(ProductID));
    REQUIRE_THROWS_AS(helper.insert(typeID, "labelA", "instanceA", "processA"), cms::Exception);
  }

  SECTION("OneEntry") {
    edm::ProductResolverIndexHelper helper;

    TypeID typeIDProductID(typeid(ProductID));
    helper.insert(typeIDProductID, "labelA", "instanceA", "processA");

    REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "processA") == ProductResolverIndexInvalid);
    REQUIRE(helper.index(ELEMENT_TYPE, typeID_ProductID, "labelA", "instanceA", "processA") == ProductResolverIndexInvalid);
    REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "") == ProductResolverIndexInvalid);
    REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA") == ProductResolverIndexInvalid);

    edm::ProductResolverIndexHelper::Matches matches = helper.relatedIndexes(PRODUCT_TYPE, typeID_ProductID, "label_A", "instance_A");
    REQUIRE(matches.numberOfMatches() == 0);
    matches = helper.relatedIndexes(PRODUCT_TYPE, typeID_ProductID);
    REQUIRE(matches.numberOfMatches() == 0);

    helper.setFrozen("processA");

    matches = helper.relatedIndexes(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA");
    REQUIRE(matches.numberOfMatches() == 2);
    edm::ProductResolverIndex indexEmptyProcess = matches.index(0);
    edm::ProductResolverIndex indexWithProcess = matches.index(1);
    REQUIRE_THROWS_AS(matches.index(2), cms::Exception);
    REQUIRE(indexEmptyProcess < 2);
    REQUIRE(indexWithProcess < 2);
    REQUIRE(indexEmptyProcess != indexWithProcess);

    REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA") == indexEmptyProcess);
    REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "") == indexEmptyProcess);
    REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", 0) == indexEmptyProcess);
    REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "processA") == indexWithProcess);

    REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instance", "processA") == ProductResolverIndexInvalid);
    REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceAX", "processA") == ProductResolverIndexInvalid);
    REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "label", "instanceA", "processA") == ProductResolverIndexInvalid);
    REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelAX", "instanceA", "processA") == ProductResolverIndexInvalid);
    REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "process") == ProductResolverIndexInvalid);
    REQUIRE(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "processAX") == ProductResolverIndexInvalid);
    REQUIRE(helper.index(PRODUCT_TYPE, typeID_EventID, "labelA", "instanceA", "processA") == ProductResolverIndexInvalid);

    REQUIRE(helper.index(ELEMENT_TYPE, typeID_ProductID, "labelA", "instanceA") == ProductResolverIndexInvalid);
    REQUIRE(helper.index(ELEMENT_TYPE, typeID_ProductID, "labelA", "instanceA", "processA") == ProductResolverIndexInvalid);

    matches = helper.relatedIndexes(PRODUCT_TYPE, typeID_ProductID);
    REQUIRE(matches.numberOfMatches() == 2);
    REQUIRE(matches.index(0) == indexEmptyProcess);
    REQUIRE(matches.index(1) == indexWithProcess);

    matches = helper.relatedIndexes(PRODUCT_TYPE, typeID_EventID);
    REQUIRE(matches.numberOfMatches() == 0);

    matches = helper.relatedIndexes(ELEMENT_TYPE, typeID_ProductID);
    REQUIRE(matches.numberOfMatches() == 0);

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

  SECTION("ManyEntries") {
    edm::ProductResolverIndexHelper helper;

    TypeID typeIDProductID(typeid(ProductID));
    TypeID typeIDEventID(typeid(EventID));
    TypeID typeIDVectorInt(typeid(std::vector<int>));
    TypeID typeIDSetInt(typeid(std::set<int>));
    TypeID typeIDVSimpleDerived(typeid(std::vector<edmtest::SimpleDerived>));

    helper.insert(typeIDVectorInt, "labelC", "instanceC", "processC");                                        // 0, 1, 2
    helper.insert(typeIDVectorInt, "label", "instance", "process");                                           // 3, 4, 5
    helper.insert(typeIDEventID, "labelB", "instanceB", "processB");                                          // 6, 7
    helper.insert(typeIDEventID, "label", "instanceB", "processB");                                           // 8, 9
    helper.insert(typeIDEventID, "labelX", "instanceB", "processB");                                          // 10, 11
    helper.insert(typeIDEventID, "labelB", "instance", "processB");                                           // 12, 13
    helper.insert(typeIDEventID, "labelB", "instanceX", "processB");                                          // 14, 15
    helper.insert(typeIDEventID, "labelB", "instanceB", "processB1");                                         // 16, 5
    helper.insert(typeIDEventID, "labelB", "instanceB", "processB3");                                         // 17, 5
    helper.insert(typeIDEventID, "labelB", "instanceB", "processB2");                                         // 18, 5
    helper.insert(typeIDProductID, "label", "instance", "process");                                           // 19, 20
    helper.insert(typeIDEventID, "label", "instance", "process");                                             // 21, 22
    helper.insert(typeIDProductID, "labelA", "instanceA", "processA");                                        // 23, 24
    REQUIRE_THROWS_AS(helper.insert(typeIDProductID, "labelA", "instanceA", "processA"), cms::Exception);  // duplicate

    helper.insert(typeIDSetInt, "labelC", "instanceC", "processC");  // 25, 26

    helper.insert(typeIDVSimpleDerived, "labelC", "instanceC", "processC");  // 27, 28, 29, 30

    helper.setFrozen("processC");

    TypeID typeID_int(typeid(int));
    REQUIRE(helper.index(ELEMENT_TYPE, typeID_int, "labelC", "instanceC", "processC") == ProductResolverIndexAmbiguous);
    REQUIRE(helper.index(ELEMENT_TYPE, typeID_int, "labelC", "instanceC", "processQ") == ProductResolverIndexInvalid);
    REQUIRE(helper.index(ELEMENT_TYPE, typeID_int, "labelC", "instanceC") == 2);
    edm::ProductResolverIndexHelper::Matches matches = helper.relatedIndexes(ELEMENT_TYPE, typeID_int);
    REQUIRE(matches.numberOfMatches() == 4);
    REQUIRE(matches.index(0) == 5);
    REQUIRE(matches.index(1) == 3);
    REQUIRE(matches.index(2) == 2);
    REQUIRE(matches.index(3) == ProductResolverIndexAmbiguous);

    TypeID typeID_vint(typeid(std::vector<int>));
    REQUIRE(helper.index(PRODUCT_TYPE, typeID_vint, "labelC", "instanceC", "processC") == 0);
    REQUIRE(helper.index(PRODUCT_TYPE, typeID_vint, "labelC", "instanceC") == 1);

    TypeID typeID_sint(typeid(std::set<int>));
    REQUIRE(helper.index(PRODUCT_TYPE, typeID_sint, "labelC", "instanceC", "processC") == 25);
    REQUIRE(helper.index(PRODUCT_TYPE, typeID_sint, "labelC", "instanceC") == 26);

    TypeID typeID_Simple(typeid(edmtest::Simple));
    REQUIRE(helper.index(ELEMENT_TYPE, typeID_Simple, "labelC", "instanceC") == 30);
    REQUIRE(helper.index(ELEMENT_TYPE, typeID_Simple, "labelC", "instanceC", "processC") == 27);

    TypeID typeID_SimpleDerived(typeid(edmtest::SimpleDerived));
    REQUIRE(helper.index(ELEMENT_TYPE, typeID_SimpleDerived, "labelC", "instanceC") == 29);
    REQUIRE(helper.index(ELEMENT_TYPE, typeID_SimpleDerived, "labelC", "instanceC", "processC") == 27);

    TypeID typeID_VSimpleDerived(typeid(std::vector<edmtest::SimpleDerived>));
    REQUIRE(helper.index(PRODUCT_TYPE, typeID_VSimpleDerived, "labelC", "instanceC") == 28);
    REQUIRE(helper.index(PRODUCT_TYPE, typeID_VSimpleDerived, "labelC", "instanceC", "processC") == 27);

    matches = helper.relatedIndexes(PRODUCT_TYPE, typeID_EventID, "labelB", "instanceB");
    REQUIRE(matches.numberOfMatches() == 5);
    ProductResolverIndex indexEmptyProcess = matches.index(0);
    ProductResolverIndex indexB = matches.index(1);
    ProductResolverIndex indexB1 = matches.index(2);
    ProductResolverIndex indexB2 = matches.index(3);
    ProductResolverIndex indexB3 = matches.index(4);
    REQUIRE_THROWS_AS(matches.index(5), cms::Exception);
    REQUIRE(indexEmptyProcess == 7);
    REQUIRE(indexB == 6);
    REQUIRE(indexB1 == 16);
    REQUIRE(indexB2 == 18);
    REQUIRE(indexB3 == 17);

    REQUIRE(std::string(matches.moduleLabel(4)) == "labelB");
    REQUIRE(std::string(matches.productInstanceName(4)) == "instanceB");
    REQUIRE(std::string(matches.processName(4)) == "processB3");
    REQUIRE(std::string(matches.processName(0)) == "");

    matches = helper.relatedIndexes(ELEMENT_TYPE, typeID_Simple);
    REQUIRE(matches.numberOfMatches() == 2);
    ProductResolverIndex indexC = matches.index(1);
    REQUIRE_THROWS_AS(matches.index(2), cms::Exception);
    REQUIRE(indexC == 27);

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
