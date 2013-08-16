/*
 *  productHolderIndexHelper_t.cppunit.cc
 */

#include "cppunit/extensions/HelperMacros.h"

#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/ProductHolderIndexHelper.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/RootAutoLibraryLoader/interface/RootAutoLibraryLoader.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/ProductKindOfType.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

#include <iostream>
#include <iomanip>

static bool alreadyCalledLoader_productHolderIndexHelper_t = false;

using namespace edm;

class TestProductHolderIndexHelper: public CppUnit::TestFixture
{
  CPPUNIT_TEST_SUITE(TestProductHolderIndexHelper);  
  CPPUNIT_TEST(testCreateEmpty);
  CPPUNIT_TEST(testOneEntry);
  CPPUNIT_TEST(testManyEntries);
  CPPUNIT_TEST_SUITE_END();
  
public:
  
  void setUp();
  void tearDown() { }

  void testCreateEmpty();
  void testOneEntry();
  void testManyEntries();

  TypeID typeID_ProductID;
  TypeID typeID_EventID;
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(TestProductHolderIndexHelper);

void TestProductHolderIndexHelper::setUp() {

  if (!alreadyCalledLoader_productHolderIndexHelper_t) {
    edm::RootAutoLibraryLoader::enable();
    alreadyCalledLoader_productHolderIndexHelper_t = true;
  }

  typeID_ProductID = TypeID(typeid(ProductID));
  typeID_EventID = TypeID(typeid(EventID));
}

void TestProductHolderIndexHelper::testCreateEmpty() {

  edm::ProductHolderIndexHelper helper;
  helper.setFrozen();

  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "processA") == ProductHolderIndexInvalid);
  CPPUNIT_ASSERT(helper.index(ELEMENT_TYPE, typeID_ProductID, "labelA", "instanceA", "processA") == ProductHolderIndexInvalid);
  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "") == ProductHolderIndexInvalid);
  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA") == ProductHolderIndexInvalid);

  edm::ProductHolderIndexHelper::Matches matches = helper.relatedIndexes(PRODUCT_TYPE, typeID_ProductID, "label_A", "instance_A");
  CPPUNIT_ASSERT(matches.numberOfMatches() == 0);
  matches = helper.relatedIndexes(PRODUCT_TYPE, typeID_ProductID);
  CPPUNIT_ASSERT(matches.numberOfMatches() == 0);

  TypeWithDict typeWithDict(typeid(ProductID));
  CPPUNIT_ASSERT_THROW( helper.insert(typeWithDict, "labelA", "instanceA", "processA") , cms::Exception);
}

void TestProductHolderIndexHelper::testOneEntry() {

  edm::ProductHolderIndexHelper helper;

  TypeWithDict typeWithDictProductID(typeid(ProductID));
  helper.insert(typeWithDictProductID, "labelA", "instanceA", "processA");

  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "processA") == ProductHolderIndexInvalid);
  CPPUNIT_ASSERT(helper.index(ELEMENT_TYPE, typeID_ProductID, "labelA", "instanceA", "processA") == ProductHolderIndexInvalid);
  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "") == ProductHolderIndexInvalid);
  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA") == ProductHolderIndexInvalid);

  edm::ProductHolderIndexHelper::Matches matches = helper.relatedIndexes(PRODUCT_TYPE, typeID_ProductID, "label_A", "instance_A");
  CPPUNIT_ASSERT(matches.numberOfMatches() == 0);
  matches = helper.relatedIndexes(PRODUCT_TYPE, typeID_ProductID);
  CPPUNIT_ASSERT(matches.numberOfMatches() == 0);

  helper.setFrozen();

  matches = helper.relatedIndexes(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA");
  CPPUNIT_ASSERT(matches.numberOfMatches() == 2);
  edm::ProductHolderIndex indexEmptyProcess = matches.index(0);
  edm::ProductHolderIndex indexWithProcess = matches.index(1);
  CPPUNIT_ASSERT_THROW(matches.index(2), cms::Exception);
  CPPUNIT_ASSERT(indexEmptyProcess < 2);
  CPPUNIT_ASSERT(indexWithProcess < 2);
  CPPUNIT_ASSERT(indexEmptyProcess != indexWithProcess);

  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA") == indexEmptyProcess);
  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "") == indexEmptyProcess);
  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", 0) == indexEmptyProcess);
  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "processA") == indexWithProcess);

  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instance", "processA") == ProductHolderIndexInvalid);
  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceAX", "processA") == ProductHolderIndexInvalid);
  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_ProductID, "label", "instanceA", "processA") == ProductHolderIndexInvalid);
  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelAX", "instanceA", "processA") == ProductHolderIndexInvalid);
  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "process") == ProductHolderIndexInvalid);
  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_ProductID, "labelA", "instanceA", "processAX") == ProductHolderIndexInvalid);
  CPPUNIT_ASSERT(helper.index(PRODUCT_TYPE, typeID_EventID, "labelA", "instanceA", "processA") == ProductHolderIndexInvalid);

  CPPUNIT_ASSERT(helper.index(ELEMENT_TYPE, typeID_ProductID, "labelA", "instanceA") == ProductHolderIndexInvalid);
  CPPUNIT_ASSERT(helper.index(ELEMENT_TYPE, typeID_ProductID, "labelA", "instanceA", "processA") == ProductHolderIndexInvalid);

  matches = helper.relatedIndexes(PRODUCT_TYPE, typeID_ProductID);
  CPPUNIT_ASSERT(matches.numberOfMatches() == 2);
  CPPUNIT_ASSERT(matches.index(0) == indexEmptyProcess);
  CPPUNIT_ASSERT(matches.index(1) == indexWithProcess);
  CPPUNIT_ASSERT(matches.isFullyResolved(0) == false);
  CPPUNIT_ASSERT(matches.isFullyResolved(1) == true);

  matches = helper.relatedIndexes(PRODUCT_TYPE, typeID_EventID);
  CPPUNIT_ASSERT(matches.numberOfMatches() == 0);

  matches = helper.relatedIndexes(ELEMENT_TYPE, typeID_ProductID);
  CPPUNIT_ASSERT(matches.numberOfMatches() == 0);

  matches = helper.relatedIndexes(ELEMENT_TYPE, typeID_ProductID, "labelA", "instanceA");
  CPPUNIT_ASSERT(matches.numberOfMatches() == 0);
}

void TestProductHolderIndexHelper::testManyEntries() {

  edm::ProductHolderIndexHelper helper;

  TypeWithDict typeWithDictProductID(typeid(ProductID));
  TypeWithDict typeWithDictEventID(typeid(EventID));
  TypeWithDict typeWithDictVectorInt(typeid(std::vector<int>));
  TypeWithDict typeWithDictSetInt(typeid(std::set<int>));
  TypeWithDict typeWithDictVSimpleDerived(typeid(std::vector<edmtest::SimpleDerived>));

  helper.insert(typeWithDictVectorInt, "labelC", "instanceC", "processC"); // 0, 1, 2
  helper.insert(typeWithDictVectorInt, "label",  "instance",  "process");  // 3, 4, 5
  helper.insert(typeWithDictEventID, "labelB", "instanceB", "processB");   // 6, 7
  helper.insert(typeWithDictEventID, "label",  "instanceB", "processB");   // 8, 9
  helper.insert(typeWithDictEventID, "labelX", "instanceB", "processB");   // 10, 11
  helper.insert(typeWithDictEventID, "labelB", "instance",  "processB");   // 12, 13
  helper.insert(typeWithDictEventID, "labelB", "instanceX", "processB");   // 14, 15
  helper.insert(typeWithDictEventID, "labelB", "instanceB", "processB1");  // 16, 5
  helper.insert(typeWithDictEventID, "labelB", "instanceB", "processB3");  // 17, 5
  helper.insert(typeWithDictEventID, "labelB", "instanceB", "processB2");  // 18, 5
  helper.insert(typeWithDictProductID, "label",  "instance",  "process");  // 19, 20
  helper.insert(typeWithDictEventID, "label",  "instance",  "process");    // 21, 22
  helper.insert(typeWithDictProductID, "labelA", "instanceA", "processA"); // 23, 24
  CPPUNIT_ASSERT_THROW(helper.insert(typeWithDictProductID, "labelA", "instanceA", "processA"), cms::Exception); // duplicate

  helper.insert(typeWithDictSetInt, "labelC", "instanceC", "processC"); // 25, 26

  helper.insert(typeWithDictVSimpleDerived, "labelC", "instanceC", "processC"); // 27, 28, 29, 30

  // helper.print(std::cout);
  helper.setFrozen();
  // helper.print(std::cout);

  TypeID typeID_int(typeid(int));
  CPPUNIT_ASSERT(helper.index(ELEMENT_TYPE, typeID_int, "labelC", "instanceC", "processC") == ProductHolderIndexAmbiguous);
  CPPUNIT_ASSERT(helper.index(ELEMENT_TYPE, typeID_int, "labelC", "instanceC", "processQ") == ProductHolderIndexInvalid);
  CPPUNIT_ASSERT(helper.index(ELEMENT_TYPE, typeID_int, "labelC", "instanceC") == 2);
  edm::ProductHolderIndexHelper::Matches matches = helper.relatedIndexes(ELEMENT_TYPE, typeID_int);
  CPPUNIT_ASSERT(matches.numberOfMatches() == 4);
  CPPUNIT_ASSERT(matches.index(0) == 5);
  CPPUNIT_ASSERT(matches.index(1) == 3);
  CPPUNIT_ASSERT(matches.index(2) == 2);
  CPPUNIT_ASSERT(matches.index(3) == ProductHolderIndexAmbiguous);

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
  ProductHolderIndex indexEmptyProcess = matches.index(0);
  ProductHolderIndex indexB = matches.index(1);
  ProductHolderIndex indexB1 = matches.index(2);
  ProductHolderIndex indexB2 = matches.index(3);
  ProductHolderIndex indexB3 = matches.index(4);
  CPPUNIT_ASSERT_THROW(matches.index(5), cms::Exception);
  CPPUNIT_ASSERT(indexEmptyProcess == 7);
  CPPUNIT_ASSERT(indexB == 6);
  CPPUNIT_ASSERT(indexB1 == 16);
  CPPUNIT_ASSERT(indexB2 == 18);
  CPPUNIT_ASSERT(indexB3 == 17);

  matches = helper.relatedIndexes(ELEMENT_TYPE, typeID_Simple);
  CPPUNIT_ASSERT(matches.numberOfMatches() == 2);
  ProductHolderIndex indexC = matches.index(1);
  CPPUNIT_ASSERT_THROW(matches.index(2), cms::Exception);
  CPPUNIT_ASSERT(indexC == 27);
}
