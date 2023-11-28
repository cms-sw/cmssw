/*
 *  productResolverIndexHelper_t.cppunit.cc
 */

#include "cppunit/extensions/HelperMacros.h"

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

class TestProductResolverIndexHelper : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestProductResolverIndexHelper);
  CPPUNIT_TEST(testCreateEmpty);
  CPPUNIT_TEST(testOneEntry);
  CPPUNIT_TEST(testManyEntries);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp();
  void tearDown() {}

  void testCreateEmpty();
  void testOneEntry();
  void testManyEntries();

  TypeID typeID_ProductID;
  TypeID typeID_EventID;
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(TestProductResolverIndexHelper);

void TestProductResolverIndexHelper::setUp() {
  typeID_ProductID = TypeID(typeid(ProductID));
  typeID_EventID = TypeID(typeid(EventID));
}

void TestProductResolverIndexHelper::testCreateEmpty() {
  edm::ProductResolverIndexHelper helper;
  helper.setFrozen();

  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "processA") ==
                 ProductResolverIndexInvalid);
  CPPUNIT_ASSERT(helper.index(ELEMENT_TYPE, typeID_ProductID, "labelA", "instanceA", "processA") ==
                 ProductResolverIndexInvalid);
  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "") ==
                 ProductResolverIndexInvalid);
  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA") == ProductResolverIndexInvalid);

  edm::ProductResolverIndexHelper::Matches matches =
      helper.relatedIndexes(PRODUCT_TYPE, typeID_ProductID, "label_A", "instance_A");
  CPPUNIT_ASSERT(matches.numberOfMatches() == 0);
  matches = helper.relatedIndexes(PRODUCT_TYPE, typeID_ProductID);
  CPPUNIT_ASSERT(matches.numberOfMatches() == 0);

  TypeID typeID(typeid(ProductID));
  CPPUNIT_ASSERT_THROW(helper.insert(typeID, "labelA", "instanceA", "processA"), cms::Exception);
}

void TestProductResolverIndexHelper::testOneEntry() {
  edm::ProductResolverIndexHelper helper;

  TypeID typeIDProductID(typeid(ProductID));
  helper.insert(typeIDProductID, "labelA", "instanceA", "processA");

  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "processA") ==
                 ProductResolverIndexInvalid);
  CPPUNIT_ASSERT(helper.index(ELEMENT_TYPE, typeID_ProductID, "labelA", "instanceA", "processA") ==
                 ProductResolverIndexInvalid);
  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "") ==
                 ProductResolverIndexInvalid);
  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA") == ProductResolverIndexInvalid);

  edm::ProductResolverIndexHelper::Matches matches =
      helper.relatedIndexes(PRODUCT_TYPE, typeID_ProductID, "label_A", "instance_A");
  CPPUNIT_ASSERT(matches.numberOfMatches() == 0);
  matches = helper.relatedIndexes(PRODUCT_TYPE, typeID_ProductID);
  CPPUNIT_ASSERT(matches.numberOfMatches() == 0);

  helper.setFrozen();

  matches = helper.relatedIndexes(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA");
  CPPUNIT_ASSERT(matches.numberOfMatches() == 2);
  edm::ProductResolverIndex indexEmptyProcess = matches.index(0);
  edm::ProductResolverIndex indexWithProcess = matches.index(1);
  CPPUNIT_ASSERT_THROW(matches.index(2), cms::Exception);
  CPPUNIT_ASSERT(indexEmptyProcess < 2);
  CPPUNIT_ASSERT(indexWithProcess < 2);
  CPPUNIT_ASSERT(indexEmptyProcess != indexWithProcess);

  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA") == indexEmptyProcess);
  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "") == indexEmptyProcess);
  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", 0) == indexEmptyProcess);
  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "processA") == indexWithProcess);

  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instance", "processA") ==
                 ProductResolverIndexInvalid);
  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceAX", "processA") ==
                 ProductResolverIndexInvalid);
  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_ProductID, "label", "instanceA", "processA") ==
                 ProductResolverIndexInvalid);
  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelAX", "instanceA", "processA") ==
                 ProductResolverIndexInvalid);
  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "process") ==
                 ProductResolverIndexInvalid);
  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "processAX") ==
                 ProductResolverIndexInvalid);
  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_EventID, "labelA", "instanceA", "processA") ==
                 ProductResolverIndexInvalid);

  CPPUNIT_ASSERT(helper.index(ELEMENT_TYPE, typeID_ProductID, "labelA", "instanceA") == ProductResolverIndexInvalid);
  CPPUNIT_ASSERT(helper.index(ELEMENT_TYPE, typeID_ProductID, "labelA", "instanceA", "processA") ==
                 ProductResolverIndexInvalid);

  matches = helper.relatedIndexes(PRODUCT_TYPE, typeID_ProductID);
  CPPUNIT_ASSERT(matches.numberOfMatches() == 2);
  CPPUNIT_ASSERT(matches.index(0) == indexEmptyProcess);
  CPPUNIT_ASSERT(matches.index(1) == indexWithProcess);

  matches = helper.relatedIndexes(PRODUCT_TYPE, typeID_EventID);
  CPPUNIT_ASSERT(matches.numberOfMatches() == 0);

  matches = helper.relatedIndexes(ELEMENT_TYPE, typeID_ProductID);
  CPPUNIT_ASSERT(matches.numberOfMatches() == 0);

  matches = helper.relatedIndexes(ELEMENT_TYPE, typeID_ProductID, "labelA", "instanceA");
  CPPUNIT_ASSERT(matches.numberOfMatches() == 0);

  {
    auto indexToModules = helper.indiciesForModulesInProcess("processA");
    CPPUNIT_ASSERT(indexToModules.size() == 1);
    CPPUNIT_ASSERT(indexToModules.count("labelA") == 1);
    auto const& range = indexToModules.equal_range("labelA");
    CPPUNIT_ASSERT(std::get<2>(range.first->second) == indexWithProcess);
  }

  {
    auto indexToModules = helper.indiciesForModulesInProcess("processNotHere");
    CPPUNIT_ASSERT(indexToModules.size() == 0);
  }
}

void TestProductResolverIndexHelper::testManyEntries() {
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
  CPPUNIT_ASSERT_THROW(helper.insert(typeIDProductID, "labelA", "instanceA", "processA"), cms::Exception);  // duplicate

  helper.insert(typeIDSetInt, "labelC", "instanceC", "processC");  // 25, 26

  helper.insert(typeIDVSimpleDerived, "labelC", "instanceC", "processC");  // 27, 28, 29, 30

  // helper.print(std::cout);
  helper.setFrozen();
  // helper.print(std::cout);

  TypeID typeID_int(typeid(int));
  CPPUNIT_ASSERT(helper.index(ELEMENT_TYPE, typeID_int, "labelC", "instanceC", "processC") ==
                 ProductResolverIndexAmbiguous);
  CPPUNIT_ASSERT(helper.index(ELEMENT_TYPE, typeID_int, "labelC", "instanceC", "processQ") ==
                 ProductResolverIndexInvalid);
  CPPUNIT_ASSERT(helper.index(ELEMENT_TYPE, typeID_int, "labelC", "instanceC") == 2);
  edm::ProductResolverIndexHelper::Matches matches = helper.relatedIndexes(ELEMENT_TYPE, typeID_int);
  CPPUNIT_ASSERT(matches.numberOfMatches() == 4);
  CPPUNIT_ASSERT(matches.index(0) == 5);
  CPPUNIT_ASSERT(matches.index(1) == 3);
  CPPUNIT_ASSERT(matches.index(2) == 2);
  CPPUNIT_ASSERT(matches.index(3) == ProductResolverIndexAmbiguous);

  TypeID typeID_vint(typeid(std::vector<int>));
  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_vint, "labelC", "instanceC", "processC") == 0);
  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_vint, "labelC", "instanceC") == 1);

  TypeID typeID_sint(typeid(std::set<int>));
  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_sint, "labelC", "instanceC", "processC") == 25);
  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_sint, "labelC", "instanceC") == 26);

  TypeID typeID_Simple(typeid(edmtest::Simple));
  CPPUNIT_ASSERT(helper.index(ELEMENT_TYPE, typeID_Simple, "labelC", "instanceC") == 30);
  CPPUNIT_ASSERT(helper.index(ELEMENT_TYPE, typeID_Simple, "labelC", "instanceC", "processC") == 27);

  TypeID typeID_SimpleDerived(typeid(edmtest::SimpleDerived));
  CPPUNIT_ASSERT(helper.index(ELEMENT_TYPE, typeID_SimpleDerived, "labelC", "instanceC") == 29);
  CPPUNIT_ASSERT(helper.index(ELEMENT_TYPE, typeID_SimpleDerived, "labelC", "instanceC", "processC") == 27);

  TypeID typeID_VSimpleDerived(typeid(std::vector<edmtest::SimpleDerived>));
  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_VSimpleDerived, "labelC", "instanceC") == 28);
  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_VSimpleDerived, "labelC", "instanceC", "processC") == 27);

  matches = helper.relatedIndexes(PRODUCT_TYPE, typeID_EventID, "labelB", "instanceB");
  CPPUNIT_ASSERT(matches.numberOfMatches() == 5);
  ProductResolverIndex indexEmptyProcess = matches.index(0);
  ProductResolverIndex indexB = matches.index(1);
  ProductResolverIndex indexB1 = matches.index(2);
  ProductResolverIndex indexB2 = matches.index(3);
  ProductResolverIndex indexB3 = matches.index(4);
  CPPUNIT_ASSERT_THROW(matches.index(5), cms::Exception);
  CPPUNIT_ASSERT(indexEmptyProcess == 7);
  CPPUNIT_ASSERT(indexB == 6);
  CPPUNIT_ASSERT(indexB1 == 16);
  CPPUNIT_ASSERT(indexB2 == 18);
  CPPUNIT_ASSERT(indexB3 == 17);

  CPPUNIT_ASSERT(std::string(matches.moduleLabel(4)) == "labelB");
  CPPUNIT_ASSERT(std::string(matches.productInstanceName(4)) == "instanceB");
  CPPUNIT_ASSERT(std::string(matches.processName(4)) == "processB3");
  CPPUNIT_ASSERT(std::string(matches.processName(0)) == "");

  matches = helper.relatedIndexes(ELEMENT_TYPE, typeID_Simple);
  CPPUNIT_ASSERT(matches.numberOfMatches() == 2);
  ProductResolverIndex indexC = matches.index(1);
  CPPUNIT_ASSERT_THROW(matches.index(2), cms::Exception);
  CPPUNIT_ASSERT(indexC == 27);

  {
    auto indexToModules = helper.indiciesForModulesInProcess("processA");
    CPPUNIT_ASSERT(indexToModules.size() == 1);
  }
  {
    auto indexToModules = helper.indiciesForModulesInProcess("processB");
    CPPUNIT_ASSERT(indexToModules.size() == 5);
  }
  {
    auto indexToModules = helper.indiciesForModulesInProcess("processB1");
    CPPUNIT_ASSERT(indexToModules.size() == 1);
  }
  {
    auto indexToModules = helper.indiciesForModulesInProcess("processB2");
    CPPUNIT_ASSERT(indexToModules.size() == 1);
  }
  {
    auto indexToModules = helper.indiciesForModulesInProcess("processB3");
    CPPUNIT_ASSERT(indexToModules.size() == 1);
  }
  {
    auto indexToModules = helper.indiciesForModulesInProcess("processC");
    CPPUNIT_ASSERT(indexToModules.size() == 3);
  }
}
