/*
 *  datakey_t.cc
 *  EDMProto
 *
 *  Created by Chris Jones on 3/31/05.
 *  Changed by Viji Sundararajan on 24-Jun-2005.
 */

#include <cppunit/extensions/HelperMacros.h>
#include <cstring>
#include "FWCore/Framework/interface/DataKey.h"
#include "FWCore/Framework/interface/HCTypeTag.h"

using namespace edm;
using namespace edm::eventsetup;

class testDataKey: public CppUnit::TestFixture {

  CPPUNIT_TEST_SUITE(testDataKey);

  CPPUNIT_TEST(nametagConstructionTest);
  CPPUNIT_TEST(nametagComparisonTest);
  CPPUNIT_TEST(nametagCopyTest);
  CPPUNIT_TEST(ConstructionTest);
  CPPUNIT_TEST(ComparisonTest);
  CPPUNIT_TEST(CopyTest);
  CPPUNIT_TEST(nocopyConstructionTest);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp(){}
  void tearDown(){}
  
  void nametagConstructionTest();
  void nametagComparisonTest();
  void nametagCopyTest();
  void ConstructionTest();
  void ComparisonTest();
  void CopyTest();
  void nocopyConstructionTest();
}; 

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testDataKey);


void testDataKey::nametagConstructionTest()
{
   const NameTag defaultTag;
   CPPUNIT_ASSERT(0 == std::strcmp("", defaultTag.value()));
   
   const NameTag namedTag("fred");
   CPPUNIT_ASSERT(0 == std::strcmp("fred", namedTag.value()));
}

void testDataKey::nametagComparisonTest()
{
   const NameTag defaultTag;
   CPPUNIT_ASSERT(defaultTag == defaultTag);
   
   const NameTag fredTag("fred");
   CPPUNIT_ASSERT(fredTag == fredTag);
   
   CPPUNIT_ASSERT(! (defaultTag == fredTag));
   
   const NameTag barneyTag("barney");
   
   CPPUNIT_ASSERT(barneyTag < fredTag);
}

void testDataKey::nametagCopyTest()
{
   const NameTag defaultTag;
   NameTag tester(defaultTag);
   CPPUNIT_ASSERT(tester == defaultTag);
   
   const NameTag fredTag("fred");
   tester = fredTag;
   CPPUNIT_ASSERT(tester == fredTag);
}

namespace datakey_t {
   class Dummy {};
   class Dummy2 {};
}
using datakey_t::Dummy;
using datakey_t::Dummy2;

namespace edm {
  namespace eventsetup {
    namespace heterocontainer {
	template<>
	const char*
	className<Dummy>() {
	   return "Dummy";
	}

	template<>
	const char*
	className<Dummy2>() {
	   return "Dummy2";
	}
    }
  }
}

void testDataKey::ConstructionTest()
{
   DataKey defaultKey;
   CPPUNIT_ASSERT(TypeTag() == defaultKey.type());
   CPPUNIT_ASSERT(0 == std::strcmp("", defaultKey.name().value()));

   DataKey dummyKey(DataKey::makeTypeTag<Dummy>(), "");
   CPPUNIT_ASSERT(DataKey::makeTypeTag<Dummy>() == dummyKey.type());
   CPPUNIT_ASSERT(0 == std::strcmp("", dummyKey.name().value()));

   DataKey namedDummyKey(DataKey::makeTypeTag<Dummy>(), "fred");
   CPPUNIT_ASSERT(DataKey::makeTypeTag<Dummy>() == namedDummyKey.type());
   CPPUNIT_ASSERT(0 == std::strcmp("fred", namedDummyKey.name().value()));
}

void testDataKey::ComparisonTest()
{
   const DataKey defaultKey;
   CPPUNIT_ASSERT(defaultKey == defaultKey);
   CPPUNIT_ASSERT(!(defaultKey < defaultKey));
   
   const DataKey dummyKey(DataKey::makeTypeTag<Dummy>(), "");
   const DataKey fredDummyKey(DataKey::makeTypeTag<Dummy>(), "fred");
   const DataKey barneyDummyKey(DataKey::makeTypeTag<Dummy>(), "barney");

   CPPUNIT_ASSERT(! (defaultKey == dummyKey));
   CPPUNIT_ASSERT(dummyKey == dummyKey);
   CPPUNIT_ASSERT(! (dummyKey == fredDummyKey));
   
   CPPUNIT_ASSERT(barneyDummyKey == barneyDummyKey);
   CPPUNIT_ASSERT(barneyDummyKey < fredDummyKey);
   CPPUNIT_ASSERT(!(fredDummyKey < barneyDummyKey));
   CPPUNIT_ASSERT(!(barneyDummyKey == fredDummyKey));
   
   const DataKey dummy2Key(DataKey::makeTypeTag<Dummy2>(), "");

   CPPUNIT_ASSERT(! (dummy2Key == dummyKey));
}

void testDataKey::CopyTest()
{
   const DataKey defaultKey;
   DataKey tester(defaultKey);
   CPPUNIT_ASSERT(tester == defaultKey);
   
   const DataKey dummyKey(DataKey::makeTypeTag<Dummy>(), "");
   tester = dummyKey;
   CPPUNIT_ASSERT(tester == dummyKey);
   const DataKey fredDummyKey(DataKey::makeTypeTag<Dummy>(), "fred");
   tester = fredDummyKey;
   CPPUNIT_ASSERT(tester == fredDummyKey);

   DataKey tester2(fredDummyKey);
   CPPUNIT_ASSERT(tester2 == fredDummyKey);
}

void testDataKey::nocopyConstructionTest()
{
   const DataKey fredDummyKey(DataKey::makeTypeTag<Dummy>(), "fred");
   const DataKey noCopyFredDummyKey(DataKey::makeTypeTag<Dummy>(), "fred", DataKey::kDoNotCopyMemory);

   CPPUNIT_ASSERT(fredDummyKey == noCopyFredDummyKey);
   
   const DataKey copyFredDummyKey(noCopyFredDummyKey);
   CPPUNIT_ASSERT(copyFredDummyKey == noCopyFredDummyKey);
   
   DataKey copy2FredDummyKey;
   copy2FredDummyKey = noCopyFredDummyKey;
   CPPUNIT_ASSERT(copy2FredDummyKey == noCopyFredDummyKey);
   
   DataKey noCopyBarneyDummyKey(DataKey::makeTypeTag<Dummy>(), "barney", DataKey::kDoNotCopyMemory);

   noCopyBarneyDummyKey = noCopyFredDummyKey;
   CPPUNIT_ASSERT(noCopyBarneyDummyKey == noCopyFredDummyKey);
}
