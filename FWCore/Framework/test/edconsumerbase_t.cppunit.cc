// -*- C++ -*-
//
// Package:     Package
// Class  :     edconsumerbase_t
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Sat, 06 Apr 2013 16:39:12 GMT
//

// system include files
#include <vector>
#include <iostream>

// user include files
#include "cppunit/extensions/HelperMacros.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ProductHolderIndexAndSkipBit.h"

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/TypeToGet.h"

#include "DataFormats/Provenance/interface/ProductHolderIndexHelper.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/RootAutoLibraryLoader/interface/RootAutoLibraryLoader.h"

#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/ProductID.h"

#include "DataFormats/Common/interface/View.h"


class TestEDConsumerBase : public CppUnit::TestFixture {
public:
  CPPUNIT_TEST_SUITE(TestEDConsumerBase);
  CPPUNIT_TEST(testRegularType);
  CPPUNIT_TEST(testViewType);
  CPPUNIT_TEST(testMany);
  CPPUNIT_TEST(testMay);
  CPPUNIT_TEST_SUITE_END();

public:
  
  void setUp();
  void tearDown() { }

  void testRegularType();
  void testViewType();
  void testMany();
  void testMay();

};

static bool s_alreadyCalledLoader = false;

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(TestEDConsumerBase);

void TestEDConsumerBase::setUp() {
  
  if (!s_alreadyCalledLoader) {
    edm::RootAutoLibraryLoader::enable();
    s_alreadyCalledLoader = true;
  }
}

namespace {
  class IntsConsumer : public edm::EDConsumerBase {
    public:
    IntsConsumer(std::vector<edm::InputTag> const& iTags) {
      m_tokens.reserve(iTags.size());
      for(auto const& tag : iTags) {
        m_tokens.push_back(consumes<std::vector<int>>(tag));
      }
    }
    
    std::vector<edm::EDGetTokenT<std::vector<int>>> m_tokens;
  };

  class IntsMayConsumer : public edm::EDConsumerBase {
  public:
    IntsMayConsumer(std::vector<edm::InputTag> const& iTags,
                    std::vector<edm::InputTag> const& iMayTags) {
      m_tokens.reserve(iTags.size());
      m_mayTokens.reserve(iMayTags.size());
      for(auto const& tag : iTags) {
        m_tokens.push_back(consumes<std::vector<int>>(tag));
      }
      for(auto const& tag : iMayTags) {
        m_mayTokens.push_back(mayConsume<std::vector<int>>(tag));
      }
    }
    
    std::vector<edm::EDGetTokenT<std::vector<int>>> m_tokens;
    std::vector<edm::EDGetTokenT<std::vector<int>>> m_mayTokens;
  };

  
  
  class TypeToGetConsumer : public edm::EDConsumerBase {
  public:
    TypeToGetConsumer(std::vector<std::pair<edm::TypeToGet,edm::InputTag>> const& iTags) {
      m_tokens.reserve(iTags.size());
      for(auto const& typeTag : iTags) {
        m_tokens.push_back(consumes(typeTag.first,typeTag.second));
      }
    }
    
    std::vector<edm::EDGetToken> m_tokens;  
  };
  
  class IntsConsumesCollectorConsumer : public edm::EDConsumerBase {
  public:
    IntsConsumesCollectorConsumer(std::vector<edm::InputTag> const& iTags) {
      m_tokens.reserve(iTags.size());
      edm::ConsumesCollector c{ consumesCollector() };
      for(auto const& tag : iTags) {
        m_tokens.push_back(c.consumes<std::vector<int>>(tag));
      }
    }
    
    std::vector<edm::EDGetTokenT<std::vector<int>>> m_tokens;
  };

}

void
TestEDConsumerBase::testRegularType()
{
  
  edm::ProductHolderIndexHelper helper;
  
  edm::TypeWithDict typeWithDictProductID(typeid(edm::ProductID));
  edm::TypeWithDict typeWithDictEventID(typeid(edm::EventID));
  edm::TypeWithDict typeWithDictVectorInt(typeid(std::vector<int>));
  edm::TypeWithDict typeWithDictSetInt(typeid(std::set<int>));
  edm::TypeWithDict typeWithDictVSimpleDerived(typeid(std::vector<edmtest::SimpleDerived>));
  
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
  helper.insert(typeWithDictSetInt, "labelC", "instanceC", "processC"); // 25, 26
  helper.insert(typeWithDictVSimpleDerived, "labelC", "instanceC", "processC"); // 27, 28, 29, 30

  helper.setFrozen();

  edm::TypeID typeID_vint(typeid(std::vector<int>));
  const auto vint_c = helper.index(edm::PRODUCT_TYPE, typeID_vint, "labelC", "instanceC", "processC");
  const auto vint_c_no_proc = helper.index(edm::PRODUCT_TYPE, typeID_vint, "labelC", "instanceC", 0);
  const auto vint_blank = helper.index(edm::PRODUCT_TYPE, typeID_vint, "label", "instance", "process");
  const auto vint_blank_no_proc = helper.index(edm::PRODUCT_TYPE, typeID_vint, "label", "instance",0);
  {
    std::vector<edm::InputTag> vTags={ {"label","instance","process"}, {"labelC","instanceC","processC"} };
    IntsConsumer intConsumer{vTags};
    intConsumer.updateLookup(edm::InEvent,helper);
  
    CPPUNIT_ASSERT(intConsumer.m_tokens[0].index()==0);
    CPPUNIT_ASSERT(intConsumer.m_tokens[1].index()==1);

    CPPUNIT_ASSERT(vint_c == intConsumer.indexFrom(intConsumer.m_tokens[1],edm::InEvent,typeID_vint).productHolderIndex());
    CPPUNIT_ASSERT(vint_blank == intConsumer.indexFrom(intConsumer.m_tokens[0],edm::InEvent,typeID_vint).productHolderIndex());
    
    std::vector<edm::ProductHolderIndexAndSkipBit> indices;
    intConsumer.itemsToGet(edm::InEvent,indices);
    
    CPPUNIT_ASSERT(2 == indices.size());
    CPPUNIT_ASSERT(indices.end() != std::find(indices.begin(),indices.end(), edm::ProductHolderIndexAndSkipBit(vint_c, false)));
    CPPUNIT_ASSERT(indices.end() != std::find(indices.begin(),indices.end(), edm::ProductHolderIndexAndSkipBit(vint_blank, false)));

    std::vector<edm::ProductHolderIndexAndSkipBit> indicesMay;
    intConsumer.itemsMayGet(edm::InEvent,indicesMay);
    CPPUNIT_ASSERT(0 == indicesMay.size());

  }
  {
    std::vector<edm::InputTag> vTags={ {"label","instance","process"}, {"labelC","instanceC","processC"} };
    IntsConsumesCollectorConsumer intConsumer{vTags};
    intConsumer.updateLookup(edm::InEvent,helper);
    
    CPPUNIT_ASSERT(intConsumer.m_tokens[0].index()==0);
    CPPUNIT_ASSERT(intConsumer.m_tokens[1].index()==1);

    CPPUNIT_ASSERT(vint_c == intConsumer.indexFrom(intConsumer.m_tokens[1],edm::InEvent,typeID_vint).productHolderIndex());
    CPPUNIT_ASSERT(vint_blank == intConsumer.indexFrom(intConsumer.m_tokens[0],edm::InEvent,typeID_vint).productHolderIndex());
    
    std::vector<edm::ProductHolderIndexAndSkipBit> indices;
    intConsumer.itemsToGet(edm::InEvent,indices);
    
    CPPUNIT_ASSERT(2 == indices.size());
    CPPUNIT_ASSERT(indices.end() != std::find(indices.begin(),indices.end(), edm::ProductHolderIndexAndSkipBit(vint_c, false)));
    CPPUNIT_ASSERT(indices.end() != std::find(indices.begin(),indices.end(), edm::ProductHolderIndexAndSkipBit(vint_blank, false)));
    
    std::vector<edm::ProductHolderIndexAndSkipBit> indicesMay;
    intConsumer.itemsMayGet(edm::InEvent,indicesMay);
    CPPUNIT_ASSERT(0 == indicesMay.size());
    
  }
  {
    std::vector<edm::InputTag> vTagsRev={ {"labelC","instanceC","processC"},{"label","instance","process"} };
    IntsConsumer intConsumerRev{vTagsRev};
    intConsumerRev.updateLookup(edm::InEvent,helper);

    CPPUNIT_ASSERT(intConsumerRev.m_tokens[0].index()==0);
    CPPUNIT_ASSERT(intConsumerRev.m_tokens[1].index()==1);

    CPPUNIT_ASSERT(vint_c == intConsumerRev.indexFrom(intConsumerRev.m_tokens[0],edm::InEvent,typeID_vint).productHolderIndex());
    CPPUNIT_ASSERT(vint_blank == intConsumerRev.indexFrom(intConsumerRev.m_tokens[1],edm::InEvent,typeID_vint).productHolderIndex());

    std::vector<edm::ProductHolderIndexAndSkipBit> indices;
    intConsumerRev.itemsToGet(edm::InEvent,indices);
    
    CPPUNIT_ASSERT(2 == indices.size());
    CPPUNIT_ASSERT(indices.end() != std::find(indices.begin(),indices.end(), edm::ProductHolderIndexAndSkipBit(vint_c, false)));
    CPPUNIT_ASSERT(indices.end() != std::find(indices.begin(),indices.end(), edm::ProductHolderIndexAndSkipBit(vint_blank, false)));
    
    std::vector<edm::ProductHolderIndexAndSkipBit> indicesMay;
    intConsumerRev.itemsMayGet(edm::InEvent,indicesMay);
    CPPUNIT_ASSERT(0 == indicesMay.size());
  }
  {
    std::vector<edm::InputTag> vTagsRev={ {"labelC","instanceC","processC"},{"label","instance","process"} };
    IntsConsumesCollectorConsumer intConsumerRev{vTagsRev};
    intConsumerRev.updateLookup(edm::InEvent,helper);
    
    CPPUNIT_ASSERT(intConsumerRev.m_tokens[0].index()==0);
    CPPUNIT_ASSERT(intConsumerRev.m_tokens[1].index()==1);

    CPPUNIT_ASSERT(edm::ProductHolderIndexAndSkipBit(vint_c, false) ==
                   intConsumerRev.indexFrom(intConsumerRev.m_tokens[0],edm::InEvent,typeID_vint));
    CPPUNIT_ASSERT(edm::ProductHolderIndexAndSkipBit(vint_blank, false) ==
                   intConsumerRev.indexFrom(intConsumerRev.m_tokens[1],edm::InEvent,typeID_vint));
    
    std::vector<edm::ProductHolderIndexAndSkipBit> indices;
    intConsumerRev.itemsToGet(edm::InEvent,indices);
    
    CPPUNIT_ASSERT(2 == indices.size());
    CPPUNIT_ASSERT(indices.end() != std::find(indices.begin(),indices.end(), edm::ProductHolderIndexAndSkipBit(vint_c, false)));
    CPPUNIT_ASSERT(indices.end() != std::find(indices.begin(),indices.end(), edm::ProductHolderIndexAndSkipBit(vint_blank, false)));
    
    std::vector<edm::ProductHolderIndexAndSkipBit> indicesMay;
    intConsumerRev.itemsMayGet(edm::InEvent,indicesMay);
    CPPUNIT_ASSERT(0 == indicesMay.size());
  }
  {
    //test default process
    std::vector<edm::InputTag> vTags={ {"label","instance"}, {"labelC","instanceC","@skipCurrentProcess"} };
    IntsConsumer intConsumer{vTags};
    intConsumer.updateLookup(edm::InEvent,helper);
    
    CPPUNIT_ASSERT(intConsumer.m_tokens[0].index()==0);
    CPPUNIT_ASSERT(intConsumer.m_tokens[1].index()==1);

    CPPUNIT_ASSERT(edm::ProductHolderIndexAndSkipBit(vint_c_no_proc, true) ==
                   intConsumer.indexFrom(intConsumer.m_tokens[1],edm::InEvent,typeID_vint));
    CPPUNIT_ASSERT(edm::ProductHolderIndexAndSkipBit(vint_blank_no_proc, false) ==
                   intConsumer.indexFrom(intConsumer.m_tokens[0],edm::InEvent,typeID_vint));

    std::vector<edm::ProductHolderIndexAndSkipBit> indices;
    intConsumer.itemsToGet(edm::InEvent,indices);
    
    CPPUNIT_ASSERT(2 == indices.size());
    CPPUNIT_ASSERT(indices.end() != std::find(indices.begin(),indices.end(), edm::ProductHolderIndexAndSkipBit(vint_c_no_proc, true)));
    CPPUNIT_ASSERT(indices.end() != std::find(indices.begin(),indices.end(), edm::ProductHolderIndexAndSkipBit(vint_blank_no_proc, false)));
    
    std::vector<edm::ProductHolderIndexAndSkipBit> indicesMay;
    intConsumer.itemsMayGet(edm::InEvent,indicesMay);
    CPPUNIT_ASSERT(0 == indicesMay.size());
  }
  {
    //Ask for something that doesn't exist
    std::vector<edm::InputTag> vTags={ {"notHere"} };
    IntsConsumer intConsumer{vTags};
    intConsumer.updateLookup(edm::InEvent,helper);
    
    CPPUNIT_ASSERT(intConsumer.m_tokens[0].index()==0);
    CPPUNIT_ASSERT(edm::ProductHolderIndexInvalid == intConsumer.indexFrom(intConsumer.m_tokens[0],edm::InEvent,typeID_vint).productHolderIndex());

    std::vector<edm::ProductHolderIndexAndSkipBit> indices;
    intConsumer.itemsToGet(edm::InEvent,indices);
    //nothing to get since not here
    CPPUNIT_ASSERT(0 == indices.size());
  }

  {
    //Use an empty tag
    std::vector<edm::InputTag> vTags={ {} };
    IntsConsumer intConsumer{vTags};
    intConsumer.updateLookup(edm::InEvent,helper);
    
    CPPUNIT_ASSERT(intConsumer.m_tokens[0].index()==0);
    CPPUNIT_ASSERT(edm::ProductHolderIndexInvalid == intConsumer.indexFrom(intConsumer.m_tokens[0],edm::InEvent,typeID_vint).productHolderIndex());
    
    std::vector<edm::ProductHolderIndexAndSkipBit> indices;
    intConsumer.itemsToGet(edm::InEvent,indices);
    //nothing to get since not here
    CPPUNIT_ASSERT(0 == indices.size());
  }
}

void
TestEDConsumerBase::testViewType()
{
  edm::ProductHolderIndexHelper helper;
  
  edm::TypeWithDict typeWithDictProductID(typeid(edm::ProductID));
  edm::TypeWithDict typeWithDictEventID(typeid(edm::EventID));
  edm::TypeWithDict typeWithDictVectorInt(typeid(std::vector<int>));
  edm::TypeWithDict typeWithDictSetInt(typeid(std::set<int>));
  edm::TypeWithDict typeWithDictVSimpleDerived(typeid(std::vector<edmtest::SimpleDerived>));
  
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
  helper.insert(typeWithDictSetInt, "labelC", "instanceC", "processC"); // 25, 26
  helper.insert(typeWithDictVSimpleDerived, "labelC", "instanceC", "processC"); // 27, 28, 29, 30
  
  helper.setFrozen();
  
  edm::TypeID typeID_int(typeid(int));
  edm::TypeID typeID_Simple(typeid(edmtest::Simple));

  const auto v_int = helper.index(edm::ELEMENT_TYPE, typeID_int, "label", "instance", "process");
  const auto v_simple = helper.index(edm::ELEMENT_TYPE, typeID_Simple, "labelC", "instanceC", "processC");

  const auto v_int_no_proc = helper.index(edm::ELEMENT_TYPE, typeID_int, "label", "instance");
  const auto v_simple_no_proc = helper.index(edm::ELEMENT_TYPE, typeID_Simple, "labelC", "instanceC");

  {
    std::vector<std::pair<edm::TypeToGet,edm::InputTag>> vT = {
      {edm::TypeToGet::make<edm::View<int>>(),{"label",  "instance",  "process"}},
      {edm::TypeToGet::make<edm::View<edmtest::Simple>>(),{"labelC", "instanceC", "processC"}}
    };
    TypeToGetConsumer consumer{vT};
    
    consumer.updateLookup(edm::InEvent,helper);
    CPPUNIT_ASSERT(edm::ProductHolderIndexAndSkipBit(v_int, false) ==
                   consumer.indexFrom(consumer.m_tokens[0],edm::InEvent,typeID_int));
    CPPUNIT_ASSERT(edm::ProductHolderIndexAndSkipBit(v_simple, false) ==
                   consumer.indexFrom(consumer.m_tokens[1],edm::InEvent,typeID_Simple));

    {
      std::vector<edm::ProductHolderIndexAndSkipBit> indices;
      consumer.itemsToGet(edm::InEvent,indices);
    
      CPPUNIT_ASSERT(2 == indices.size());
      CPPUNIT_ASSERT(indices.end() != std::find(indices.begin(),indices.end(), edm::ProductHolderIndexAndSkipBit(v_int, false)));
      CPPUNIT_ASSERT(indices.end() != std::find(indices.begin(),indices.end(), edm::ProductHolderIndexAndSkipBit(v_simple, false)));

      std::vector<edm::ProductHolderIndexAndSkipBit> indicesMay;
      consumer.itemsMayGet(edm::InEvent,indicesMay);
      CPPUNIT_ASSERT(0 == indicesMay.size());
    }
  }

  {
    std::vector<std::pair<edm::TypeToGet,edm::InputTag>> vT = {
      {edm::TypeToGet::make<edm::View<int>>(),{"label",  "instance"}},
      {edm::TypeToGet::make<edm::View<edmtest::Simple>>(),{"labelC", "instanceC","@skipCurrentProcess"}}
    };
    TypeToGetConsumer consumer{vT};

    consumer.updateLookup(edm::InEvent,helper);

    CPPUNIT_ASSERT(edm::ProductHolderIndexAndSkipBit(v_int_no_proc, false) ==
                   consumer.indexFrom(consumer.m_tokens[0],edm::InEvent,typeID_int));
    CPPUNIT_ASSERT(edm::ProductHolderIndexAndSkipBit(v_simple_no_proc, true) ==
                   consumer.indexFrom(consumer.m_tokens[1],edm::InEvent,typeID_Simple));
    {
      std::vector<edm::ProductHolderIndexAndSkipBit> indices;
      consumer.itemsToGet(edm::InEvent,indices);
      
      CPPUNIT_ASSERT(2 == indices.size());
      CPPUNIT_ASSERT(indices.end() != std::find(indices.begin(),indices.end(), edm::ProductHolderIndexAndSkipBit(v_int_no_proc, false)));
      CPPUNIT_ASSERT(indices.end() != std::find(indices.begin(),indices.end(), edm::ProductHolderIndexAndSkipBit(v_simple_no_proc, true)));
      
      std::vector<edm::ProductHolderIndexAndSkipBit> indicesMay;
      consumer.itemsMayGet(edm::InEvent,indicesMay);
      CPPUNIT_ASSERT(0 == indicesMay.size());
    }
  }

  {
    //Ask for something that doesn't exist
    std::vector<std::pair<edm::TypeToGet,edm::InputTag>> vT = {
      {edm::TypeToGet::make<edm::View<int>>(),{"notHere",  ""}}
    };
    TypeToGetConsumer consumer{vT};
    consumer.updateLookup(edm::InEvent,helper);
    
    CPPUNIT_ASSERT(consumer.m_tokens[0].index()==0);
    CPPUNIT_ASSERT(edm::ProductHolderIndexInvalid == consumer.indexFrom(consumer.m_tokens[0],edm::InEvent,typeID_int).productHolderIndex());
    {
      std::vector<edm::ProductHolderIndexAndSkipBit> indices;
      consumer.itemsToGet(edm::InEvent,indices);
      
      CPPUNIT_ASSERT(0 == indices.size());
    }
  }
}

namespace {
  class ManyEventIDConsumer : public edm::EDConsumerBase {
  public:
    ManyEventIDConsumer() {
      consumesMany<edm::EventID>();
    }
  };
}

void
TestEDConsumerBase::testMany()
{
  
  edm::ProductHolderIndexHelper helper;
  
  edm::TypeWithDict typeWithDictProductID(typeid(edm::ProductID));
  edm::TypeWithDict typeWithDictEventID(typeid(edm::EventID));
  edm::TypeWithDict typeWithDictVectorInt(typeid(std::vector<int>));
  edm::TypeWithDict typeWithDictSetInt(typeid(std::set<int>));
  edm::TypeWithDict typeWithDictVSimpleDerived(typeid(std::vector<edmtest::SimpleDerived>));
  
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
  helper.insert(typeWithDictSetInt, "labelC", "instanceC", "processC"); // 25, 26
  helper.insert(typeWithDictVSimpleDerived, "labelC", "instanceC", "processC"); // 27, 28, 29, 30
  
  helper.setFrozen();

  edm::TypeID typeID_EventID(typeid(edm::EventID));

  const auto productIndex = helper.index(edm::PRODUCT_TYPE, typeID_EventID, "labelB", "instanceB", "processB");

  {
    ManyEventIDConsumer consumer{};
    consumer.updateLookup(edm::InEvent,helper);

    std::vector<edm::ProductHolderIndexAndSkipBit> indices;
    consumer.itemsToGet(edm::InEvent,indices);

    CPPUNIT_ASSERT(9 == indices.size());
    CPPUNIT_ASSERT(indices.end() != std::find(indices.begin(),indices.end(), edm::ProductHolderIndexAndSkipBit(productIndex, false)));
  }
}

void
TestEDConsumerBase::testMay()
{
 
  edm::ProductHolderIndexHelper helper;
  
  edm::TypeWithDict typeWithDictProductID(typeid(edm::ProductID));
  edm::TypeWithDict typeWithDictEventID(typeid(edm::EventID));
  edm::TypeWithDict typeWithDictVectorInt(typeid(std::vector<int>));
  edm::TypeWithDict typeWithDictSetInt(typeid(std::set<int>));
  edm::TypeWithDict typeWithDictVSimpleDerived(typeid(std::vector<edmtest::SimpleDerived>));
  
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
  helper.insert(typeWithDictSetInt, "labelC", "instanceC", "processC"); // 25, 26
  helper.insert(typeWithDictVSimpleDerived, "labelC", "instanceC", "processC"); // 27, 28, 29, 30
  
  helper.setFrozen();
  edm::TypeID typeID_vint(typeid(std::vector<int>));
  const auto vint_c = helper.index(edm::PRODUCT_TYPE, typeID_vint, "labelC", "instanceC", "processC");
  const auto vint_c_no_proc = helper.index(edm::PRODUCT_TYPE, typeID_vint, "labelC", "instanceC", 0);
  const auto vint_blank = helper.index(edm::PRODUCT_TYPE, typeID_vint, "label", "instance", "process");
  const auto vint_blank_no_proc = helper.index(edm::PRODUCT_TYPE, typeID_vint, "label", "instance",0);
  {
    std::vector<edm::InputTag> vTags={ {"label","instance","process"}, {"labelC","instanceC","processC"} };
    std::vector<edm::InputTag> vMayTags={};
    IntsMayConsumer consumer{vTags,vMayTags};
    consumer.updateLookup(edm::InEvent,helper);
    
    CPPUNIT_ASSERT(consumer.m_tokens[0].index()==0);
    CPPUNIT_ASSERT(consumer.m_tokens[1].index()==1);
    CPPUNIT_ASSERT(consumer.m_mayTokens.size()==0);

    CPPUNIT_ASSERT(edm::ProductHolderIndexAndSkipBit(vint_c, false) ==
                   consumer.indexFrom(consumer.m_tokens[1],edm::InEvent,typeID_vint));
    CPPUNIT_ASSERT(edm::ProductHolderIndexAndSkipBit(vint_blank, false) ==
                   consumer.indexFrom(consumer.m_tokens[0],edm::InEvent,typeID_vint));

    std::vector<edm::ProductHolderIndexAndSkipBit> indices;
    consumer.itemsToGet(edm::InEvent,indices);
    
    CPPUNIT_ASSERT(2 == indices.size());
    CPPUNIT_ASSERT(indices.end() != std::find(indices.begin(),indices.end(), edm::ProductHolderIndexAndSkipBit(vint_c, false)));
    CPPUNIT_ASSERT(indices.end() != std::find(indices.begin(),indices.end(), edm::ProductHolderIndexAndSkipBit(vint_blank, false)));
    
    std::vector<edm::ProductHolderIndexAndSkipBit> indicesMay;
    consumer.itemsMayGet(edm::InEvent,indicesMay);
    CPPUNIT_ASSERT(0 == indicesMay.size());
  }

  {
    std::vector<edm::InputTag> vTags={};
    std::vector<edm::InputTag> vMayTags={ {"label","instance","process"}, {"labelC","instanceC","processC"} };
    IntsMayConsumer consumer{vTags,vMayTags};
    consumer.updateLookup(edm::InEvent,helper);
    
    CPPUNIT_ASSERT(consumer.m_mayTokens.size()==2);
    CPPUNIT_ASSERT(consumer.m_mayTokens[0].index()==0);
    CPPUNIT_ASSERT(consumer.m_mayTokens[1].index()==1);
    CPPUNIT_ASSERT(consumer.m_tokens.size()==0);

    CPPUNIT_ASSERT(vint_c == consumer.indexFrom(consumer.m_mayTokens[1],edm::InEvent,typeID_vint).productHolderIndex());
    CPPUNIT_ASSERT(vint_blank == consumer.indexFrom(consumer.m_mayTokens[0],edm::InEvent,typeID_vint).productHolderIndex());
    
    std::vector<edm::ProductHolderIndexAndSkipBit> indices;
    consumer.itemsToGet(edm::InEvent,indices);
    CPPUNIT_ASSERT(0 == indices.size());
    
    std::vector<edm::ProductHolderIndexAndSkipBit> indicesMay;
    consumer.itemsMayGet(edm::InEvent,indicesMay);
    CPPUNIT_ASSERT(2 == indicesMay.size());
    CPPUNIT_ASSERT(indicesMay.end() != std::find(indicesMay.begin(),indicesMay.end(), edm::ProductHolderIndexAndSkipBit(vint_c, false)));
    CPPUNIT_ASSERT(indicesMay.end() != std::find(indicesMay.begin(),indicesMay.end(), edm::ProductHolderIndexAndSkipBit(vint_blank, false)));
  }

  {
    std::vector<edm::InputTag> vTags={ {"label","instance","process"} };
    std::vector<edm::InputTag> vMayTags={{"labelC","instanceC","processC"} };
    IntsMayConsumer consumer{vTags,vMayTags};
    consumer.updateLookup(edm::InEvent,helper);
    
    CPPUNIT_ASSERT(consumer.m_mayTokens.size()==1);
    CPPUNIT_ASSERT(consumer.m_tokens.size()==1);
    CPPUNIT_ASSERT(consumer.m_tokens[0].index()==0);
    CPPUNIT_ASSERT(consumer.m_mayTokens[0].index()==1);

    CPPUNIT_ASSERT(vint_c == consumer.indexFrom(consumer.m_mayTokens[0],edm::InEvent,typeID_vint).productHolderIndex());
    CPPUNIT_ASSERT(vint_blank == consumer.indexFrom(consumer.m_tokens[0],edm::InEvent,typeID_vint).productHolderIndex());
    
    std::vector<edm::ProductHolderIndexAndSkipBit> indices;
    consumer.itemsToGet(edm::InEvent,indices);
    
    CPPUNIT_ASSERT(1 == indices.size());
    CPPUNIT_ASSERT(indices.end() != std::find(indices.begin(),indices.end(), edm::ProductHolderIndexAndSkipBit(vint_blank, false)));
    
    std::vector<edm::ProductHolderIndexAndSkipBit> indicesMay;
    consumer.itemsMayGet(edm::InEvent,indicesMay);
    CPPUNIT_ASSERT(1 == indicesMay.size());
    CPPUNIT_ASSERT(indicesMay.end() != std::find(indicesMay.begin(),indicesMay.end(), edm::ProductHolderIndexAndSkipBit(vint_c, false)));
  }
  {
    std::vector<edm::InputTag> vTags={};
    std::vector<edm::InputTag> vMayTags={ {"label","instance",""}, {"labelC","instanceC","@skipCurrentProcess"} };
    IntsMayConsumer consumer{vTags,vMayTags};
    consumer.updateLookup(edm::InEvent,helper);
    
    CPPUNIT_ASSERT(consumer.m_mayTokens.size()==2);
    CPPUNIT_ASSERT(consumer.m_mayTokens[0].index()==0);
    CPPUNIT_ASSERT(consumer.m_mayTokens[1].index()==1);
    CPPUNIT_ASSERT(consumer.m_tokens.size()==0);

    CPPUNIT_ASSERT(edm::ProductHolderIndexAndSkipBit(vint_c_no_proc, true) ==
                   consumer.indexFrom(consumer.m_mayTokens[1],edm::InEvent,typeID_vint));
    CPPUNIT_ASSERT(edm::ProductHolderIndexAndSkipBit(vint_blank_no_proc, false) ==
                   consumer.indexFrom(consumer.m_mayTokens[0],edm::InEvent,typeID_vint));
  }
}
