/*
 *  eventsetupscontroller_t.cc
 */

#include "cppunit/extensions/HelperMacros.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "FWCore/Framework/interface/EventSetupProvider.h"
#include "FWCore/Framework/interface/EventSetupsController.h"
#include "FWCore/Framework/interface/ParameterSetIDHolder.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/test/DummyFinder.h"
#include "FWCore/Framework/test/DummyESProductResolverProvider.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <memory>
#include <string>
#include <vector>

namespace {
  edm::ActivityRegistry activityRegistry;
}

class TestEventSetupsController : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(TestEventSetupsController);

  CPPUNIT_TEST(constructorTest);
  CPPUNIT_TEST(esProducerGetAndPutTest);
  CPPUNIT_TEST(esSourceGetAndPutTest);

  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}

  void constructorTest();
  void esProducerGetAndPutTest();
  void esSourceGetAndPutTest();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(TestEventSetupsController);

void TestEventSetupsController::constructorTest() {
  edm::eventsetup::EventSetupsController esController;

  CPPUNIT_ASSERT(esController.providers().empty());
  CPPUNIT_ASSERT(esController.esproducers().empty());
  CPPUNIT_ASSERT(esController.essources().empty());
  CPPUNIT_ASSERT(esController.mustFinishConfiguration() == true);

  edm::ParameterSet pset;
  std::vector<std::string> emptyVStrings;
  pset.addParameter<std::vector<std::string> >("@all_esprefers", emptyVStrings);
  pset.addParameter<std::vector<std::string> >("@all_essources", emptyVStrings);
  pset.addParameter<std::vector<std::string> >("@all_esmodules", emptyVStrings);

  esController.makeProvider(pset, &activityRegistry);
  esController.makeProvider(pset, &activityRegistry);
  esController.makeProvider(pset, &activityRegistry);

  CPPUNIT_ASSERT(esController.providers().size() == 3);
  CPPUNIT_ASSERT(esController.providers()[0]->subProcessIndex() == 0);
  CPPUNIT_ASSERT(esController.providers()[1]->subProcessIndex() == 1);
  CPPUNIT_ASSERT(esController.providers()[2]->subProcessIndex() == 2);
}

void TestEventSetupsController::esProducerGetAndPutTest() {
  edm::eventsetup::EventSetupsController esController;

  edm::ParameterSet pset1;
  pset1.registerIt();
  std::shared_ptr<edm::eventsetup::test::DummyESProductResolverProvider> resolverProvider1 =
      std::make_shared<edm::eventsetup::test::DummyESProductResolverProvider>();

  edm::ParameterSet pset2;
  pset2.addUntrackedParameter<int>("p1", 1);
  pset2.registerIt();
  std::shared_ptr<edm::eventsetup::test::DummyESProductResolverProvider> resolverProvider2 =
      std::make_shared<edm::eventsetup::test::DummyESProductResolverProvider>();
  CPPUNIT_ASSERT(pset2.id() == pset1.id());

  edm::ParameterSet pset3;
  pset3.addUntrackedParameter<int>("p1", 2);
  pset3.registerIt();
  std::shared_ptr<edm::eventsetup::test::DummyESProductResolverProvider> resolverProvider3 =
      std::make_shared<edm::eventsetup::test::DummyESProductResolverProvider>();
  CPPUNIT_ASSERT(pset3.id() == pset1.id());

  edm::ParameterSet pset4;
  pset4.addParameter<int>("p1", 1);
  pset4.registerIt();
  std::shared_ptr<edm::eventsetup::test::DummyESProductResolverProvider> resolverProvider4 =
      std::make_shared<edm::eventsetup::test::DummyESProductResolverProvider>();
  CPPUNIT_ASSERT(pset4.id() != pset1.id());

  edm::eventsetup::ParameterSetIDHolder psetIDHolder1(pset1.id());
  edm::eventsetup::ParameterSetIDHolder psetIDHolder1a(pset1.id());
  edm::eventsetup::ParameterSetIDHolder psetIDHolder4(pset4.id());
  CPPUNIT_ASSERT(pset1.id() == psetIDHolder1.psetID());
  CPPUNIT_ASSERT(psetIDHolder1 == psetIDHolder1a);
  CPPUNIT_ASSERT(!(psetIDHolder1 == psetIDHolder4));
  CPPUNIT_ASSERT((pset1.id() < pset4.id()) == (psetIDHolder1 < psetIDHolder4));

  std::shared_ptr<edm::eventsetup::ESProductResolverProvider> ptrFromGet =
      esController.getESProducerAndRegisterProcess(pset1, 0);
  CPPUNIT_ASSERT(!ptrFromGet);
  esController.putESProducer(pset1, resolverProvider1, 0);

  ptrFromGet = esController.getESProducerAndRegisterProcess(pset2, 0);
  CPPUNIT_ASSERT(!ptrFromGet);
  esController.putESProducer(pset2, resolverProvider2, 0);

  ptrFromGet = esController.getESProducerAndRegisterProcess(pset3, 0);
  CPPUNIT_ASSERT(!ptrFromGet);
  esController.putESProducer(pset3, resolverProvider3, 0);

  ptrFromGet = esController.getESProducerAndRegisterProcess(pset4, 0);
  CPPUNIT_ASSERT(!ptrFromGet);
  esController.putESProducer(pset4, resolverProvider4, 0);

  ptrFromGet = esController.getESProducerAndRegisterProcess(pset1, 1);
  CPPUNIT_ASSERT(ptrFromGet);
  CPPUNIT_ASSERT(ptrFromGet == resolverProvider1);

  ptrFromGet = esController.getESProducerAndRegisterProcess(pset2, 2);
  CPPUNIT_ASSERT(ptrFromGet);
  CPPUNIT_ASSERT(ptrFromGet == resolverProvider2);

  ptrFromGet = esController.getESProducerAndRegisterProcess(pset3, 3);
  CPPUNIT_ASSERT(ptrFromGet);
  CPPUNIT_ASSERT(ptrFromGet == resolverProvider3);

  ptrFromGet = esController.getESProducerAndRegisterProcess(pset4, 4);
  CPPUNIT_ASSERT(ptrFromGet);
  CPPUNIT_ASSERT(ptrFromGet == resolverProvider4);

  ptrFromGet = esController.getESProducerAndRegisterProcess(pset4, 5);
  CPPUNIT_ASSERT(ptrFromGet);
  CPPUNIT_ASSERT(ptrFromGet == resolverProvider4);

  std::multimap<edm::ParameterSetID, edm::eventsetup::ESProducerInfo> const& esproducers = esController.esproducers();
  bool isPresent1 = false;
  bool isPresent2 = false;
  bool isPresent3 = false;
  bool isPresent4 = false;

  CPPUNIT_ASSERT(esproducers.size() == 4);
  for (auto const& esproducer : esproducers) {
    if (esproducer.second.pset() == &pset1) {
      isPresent1 = true;
      CPPUNIT_ASSERT(esproducer.first == pset1.id());
      CPPUNIT_ASSERT(esproducer.second.providerGet() ==
                     static_cast<edm::eventsetup::ESProductResolverProvider*>(resolverProvider1.get()));
      edm::eventsetup::ESProducerInfo const& info = esproducer.second;
      CPPUNIT_ASSERT(info.subProcessIndexes().size() == 2);
      CPPUNIT_ASSERT(info.subProcessIndexes()[0] == 0);
      CPPUNIT_ASSERT(info.subProcessIndexes()[1] == 1);
    }
    if (esproducer.second.pset() == &pset2) {
      isPresent2 = true;
      CPPUNIT_ASSERT(esproducer.first == pset1.id());
      CPPUNIT_ASSERT(esproducer.second.providerGet() ==
                     static_cast<edm::eventsetup::ESProductResolverProvider*>(resolverProvider2.get()));
      edm::eventsetup::ESProducerInfo const& info = esproducer.second;
      CPPUNIT_ASSERT(info.subProcessIndexes().size() == 2);
      CPPUNIT_ASSERT(info.subProcessIndexes()[0] == 0);
      CPPUNIT_ASSERT(info.subProcessIndexes()[1] == 2);
    }
    if (esproducer.second.pset() == &pset3) {
      isPresent3 = true;
      CPPUNIT_ASSERT(esproducer.first == pset3.id());
      CPPUNIT_ASSERT(esproducer.second.providerGet() ==
                     static_cast<edm::eventsetup::ESProductResolverProvider*>(resolverProvider3.get()));
      edm::eventsetup::ESProducerInfo const& info = esproducer.second;
      CPPUNIT_ASSERT(info.subProcessIndexes().size() == 2);
      CPPUNIT_ASSERT(info.subProcessIndexes()[0] == 0);
      CPPUNIT_ASSERT(info.subProcessIndexes()[1] == 3);
    }
    if (esproducer.second.pset() == &pset4) {
      isPresent4 = true;
      CPPUNIT_ASSERT(esproducer.first == pset4.id());
      CPPUNIT_ASSERT(esproducer.second.providerGet() ==
                     static_cast<edm::eventsetup::ESProductResolverProvider*>(resolverProvider4.get()));
      edm::eventsetup::ESProducerInfo const& info = esproducer.second;
      CPPUNIT_ASSERT(info.subProcessIndexes().size() == 3);
      CPPUNIT_ASSERT(info.subProcessIndexes()[0] == 0);
      CPPUNIT_ASSERT(info.subProcessIndexes()[1] == 4);
      CPPUNIT_ASSERT(info.subProcessIndexes()[2] == 5);
    }
  }
  CPPUNIT_ASSERT(isPresent1 && isPresent2 && isPresent3 && isPresent4);

  bool firstProcessWithThisPSet = false;
  bool precedingHasMatchingPSet = false;

  esController.lookForMatches(pset4.id(), 0, 0, firstProcessWithThisPSet, precedingHasMatchingPSet);
  CPPUNIT_ASSERT(firstProcessWithThisPSet == true);
  CPPUNIT_ASSERT(precedingHasMatchingPSet == false);

  esController.lookForMatches(pset4.id(), 4, 0, firstProcessWithThisPSet, precedingHasMatchingPSet);
  CPPUNIT_ASSERT(firstProcessWithThisPSet == false);
  CPPUNIT_ASSERT(precedingHasMatchingPSet == true);

  esController.lookForMatches(pset4.id(), 4, 1, firstProcessWithThisPSet, precedingHasMatchingPSet);
  CPPUNIT_ASSERT(firstProcessWithThisPSet == false);
  CPPUNIT_ASSERT(precedingHasMatchingPSet == false);

  CPPUNIT_ASSERT_THROW(
      esController.lookForMatches(pset4.id(), 6, 0, firstProcessWithThisPSet, precedingHasMatchingPSet),
      cms::Exception);

  CPPUNIT_ASSERT(esController.isFirstMatch(pset4.id(), 5, 0));
  CPPUNIT_ASSERT(!esController.isFirstMatch(pset4.id(), 5, 4));
  CPPUNIT_ASSERT_THROW(esController.isFirstMatch(pset4.id(), 6, 4), cms::Exception);
  CPPUNIT_ASSERT_THROW(esController.isFirstMatch(pset4.id(), 5, 1), cms::Exception);

  CPPUNIT_ASSERT(!esController.isLastMatch(pset4.id(), 5, 0));
  CPPUNIT_ASSERT(esController.isLastMatch(pset4.id(), 5, 4));
  CPPUNIT_ASSERT_THROW(esController.isLastMatch(pset4.id(), 6, 4), cms::Exception);
  CPPUNIT_ASSERT_THROW(esController.isLastMatch(pset4.id(), 5, 1), cms::Exception);

  CPPUNIT_ASSERT(esController.isMatchingESProducer(pset4.id(), 5, 0));
  CPPUNIT_ASSERT(esController.isMatchingESProducer(pset4.id(), 5, 4));
  CPPUNIT_ASSERT(!esController.isMatchingESProducer(pset4.id(), 5, 2));
  CPPUNIT_ASSERT_THROW(esController.isMatchingESProducer(pset4.id(), 6, 4), cms::Exception);

  CPPUNIT_ASSERT(&esController.getESProducerPSet(pset1.id(), 0) == &pset1);
  CPPUNIT_ASSERT(&esController.getESProducerPSet(pset2.id(), 2) == &pset2);
  CPPUNIT_ASSERT(&esController.getESProducerPSet(pset3.id(), 3) == &pset3);
  CPPUNIT_ASSERT(&esController.getESProducerPSet(pset4.id(), 5) == &pset4);
  CPPUNIT_ASSERT_THROW(esController.getESProducerPSet(pset4.id(), 6), cms::Exception);

  esController.clearComponents();
  CPPUNIT_ASSERT(esController.esproducers().empty());
  CPPUNIT_ASSERT(esController.essources().empty());
}

void TestEventSetupsController::esSourceGetAndPutTest() {
  edm::eventsetup::EventSetupsController esController;

  edm::ParameterSet pset1;
  pset1.registerIt();
  std::shared_ptr<DummyFinder> finder1 = std::make_shared<DummyFinder>();

  edm::ParameterSet pset2;
  pset2.addUntrackedParameter<int>("p1", 1);
  pset2.registerIt();
  std::shared_ptr<DummyFinder> finder2 = std::make_shared<DummyFinder>();
  CPPUNIT_ASSERT(pset2.id() == pset1.id());

  edm::ParameterSet pset3;
  pset3.addUntrackedParameter<int>("p1", 2);
  pset3.registerIt();
  std::shared_ptr<DummyFinder> finder3 = std::make_shared<DummyFinder>();
  CPPUNIT_ASSERT(pset3.id() == pset1.id());

  edm::ParameterSet pset4;
  pset4.addParameter<int>("p1", 1);
  pset4.registerIt();
  std::shared_ptr<DummyFinder> finder4 = std::make_shared<DummyFinder>();
  CPPUNIT_ASSERT(pset4.id() != pset1.id());

  std::shared_ptr<edm::EventSetupRecordIntervalFinder> ptrFromGet =
      esController.getESSourceAndRegisterProcess(pset1, 0);
  CPPUNIT_ASSERT(!ptrFromGet);
  esController.putESSource(pset1, finder1, 0);

  ptrFromGet = esController.getESSourceAndRegisterProcess(pset2, 0);
  CPPUNIT_ASSERT(!ptrFromGet);
  esController.putESSource(pset2, finder2, 0);

  ptrFromGet = esController.getESSourceAndRegisterProcess(pset3, 0);
  CPPUNIT_ASSERT(!ptrFromGet);
  esController.putESSource(pset3, finder3, 0);

  ptrFromGet = esController.getESSourceAndRegisterProcess(pset4, 0);
  CPPUNIT_ASSERT(!ptrFromGet);
  esController.putESSource(pset4, finder4, 0);

  ptrFromGet = esController.getESSourceAndRegisterProcess(pset1, 1);
  CPPUNIT_ASSERT(ptrFromGet);
  CPPUNIT_ASSERT(ptrFromGet == finder1);

  ptrFromGet = esController.getESSourceAndRegisterProcess(pset2, 2);
  CPPUNIT_ASSERT(ptrFromGet);
  CPPUNIT_ASSERT(ptrFromGet == finder2);

  ptrFromGet = esController.getESSourceAndRegisterProcess(pset3, 3);
  CPPUNIT_ASSERT(ptrFromGet);
  CPPUNIT_ASSERT(ptrFromGet == finder3);

  ptrFromGet = esController.getESSourceAndRegisterProcess(pset4, 4);
  CPPUNIT_ASSERT(ptrFromGet);
  CPPUNIT_ASSERT(ptrFromGet == finder4);

  ptrFromGet = esController.getESSourceAndRegisterProcess(pset4, 5);
  CPPUNIT_ASSERT(ptrFromGet);
  CPPUNIT_ASSERT(ptrFromGet == finder4);

  std::multimap<edm::ParameterSetID, edm::eventsetup::ESSourceInfo> const& essources = esController.essources();
  bool isPresent1 = false;
  bool isPresent2 = false;
  bool isPresent3 = false;
  bool isPresent4 = false;

  CPPUNIT_ASSERT(essources.size() == 4);
  for (auto const& essource : essources) {
    if (essource.second.pset() == &pset1) {
      isPresent1 = true;
      CPPUNIT_ASSERT(essource.first == pset1.id());
      CPPUNIT_ASSERT(essource.second.finderGet() ==
                     static_cast<edm::EventSetupRecordIntervalFinder const*>(finder1.get()));
      edm::eventsetup::ESSourceInfo const& info = essource.second;
      CPPUNIT_ASSERT(info.subProcessIndexes().size() == 2);
      CPPUNIT_ASSERT(info.subProcessIndexes()[0] == 0);
      CPPUNIT_ASSERT(info.subProcessIndexes()[1] == 1);
    }
    if (essource.second.pset() == &pset2) {
      isPresent2 = true;
      CPPUNIT_ASSERT(essource.first == pset1.id());
      CPPUNIT_ASSERT(essource.second.finderGet() ==
                     static_cast<edm::EventSetupRecordIntervalFinder const*>(finder2.get()));
      edm::eventsetup::ESSourceInfo const& info = essource.second;
      CPPUNIT_ASSERT(info.subProcessIndexes().size() == 2);
      CPPUNIT_ASSERT(info.subProcessIndexes()[0] == 0);
      CPPUNIT_ASSERT(info.subProcessIndexes()[1] == 2);
    }
    if (essource.second.pset() == &pset3) {
      isPresent3 = true;
      CPPUNIT_ASSERT(essource.first == pset3.id());
      CPPUNIT_ASSERT(essource.second.finderGet() ==
                     static_cast<edm::EventSetupRecordIntervalFinder const*>(finder3.get()));
      edm::eventsetup::ESSourceInfo const& info = essource.second;
      CPPUNIT_ASSERT(info.subProcessIndexes().size() == 2);
      CPPUNIT_ASSERT(info.subProcessIndexes()[0] == 0);
      CPPUNIT_ASSERT(info.subProcessIndexes()[1] == 3);
    }
    if (essource.second.pset() == &pset4) {
      isPresent4 = true;
      CPPUNIT_ASSERT(essource.first == pset4.id());
      CPPUNIT_ASSERT(essource.second.finderGet() ==
                     static_cast<edm::EventSetupRecordIntervalFinder const*>(finder4.get()));
      edm::eventsetup::ESSourceInfo const& info = essource.second;
      CPPUNIT_ASSERT(info.subProcessIndexes().size() == 3);
      CPPUNIT_ASSERT(info.subProcessIndexes()[0] == 0);
      CPPUNIT_ASSERT(info.subProcessIndexes()[1] == 4);
      CPPUNIT_ASSERT(info.subProcessIndexes()[2] == 5);
    }
  }
  CPPUNIT_ASSERT(isPresent1 && isPresent2 && isPresent3 && isPresent4);

  CPPUNIT_ASSERT(esController.isMatchingESSource(pset4.id(), 5, 0));
  CPPUNIT_ASSERT(esController.isMatchingESSource(pset4.id(), 5, 4));
  CPPUNIT_ASSERT(!esController.isMatchingESSource(pset4.id(), 5, 2));
  CPPUNIT_ASSERT_THROW(esController.isMatchingESSource(pset4.id(), 6, 4), cms::Exception);

  esController.clearComponents();
  CPPUNIT_ASSERT(esController.esproducers().empty());
  CPPUNIT_ASSERT(esController.essources().empty());
}
