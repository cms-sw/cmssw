/*----------------------------------------------------------------------

Test of the EventPrincipal class.

----------------------------------------------------------------------*/
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ProductDescription.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"

#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/HistoryAppender.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/ProducesCollector.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/ProducerBase.h"
#include "FWCore/Framework/interface/ProductResolversFactory.h"
#include "FWCore/Framework/interface/SignallingProductRegistryFiller.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/GlobalIdentifier.h"
#include "FWCore/Reflection/interface/TypeWithDict.h"

#include "cppunit/extensions/HelperMacros.h"

#include <cassert>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeinfo>

#include "FWCore/Framework/interface/Event.h"

#include "makeDummyProcessConfiguration.h"

class testEventGetRefBeforePut : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testEventGetRefBeforePut);
  CPPUNIT_TEST(failGetProductNotRegisteredTest);
  CPPUNIT_TEST(getRefTest);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}
  void failGetProductNotRegisteredTest();
  void getRefTest();

private:
  edm::HistoryAppender historyAppender_;
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testEventGetRefBeforePut);

namespace {
  class TestProducer : public edm::ProducerBase {
  public:
    TestProducer(std::string const& productInstanceName) { produces<edmtest::IntProduct>(productInstanceName); }
  };
}  // namespace

void testEventGetRefBeforePut::failGetProductNotRegisteredTest() {
  auto preg = std::make_unique<edm::ProductRegistry>();
  preg->setFrozen();
  auto branchIDListHelper = std::make_shared<edm::BranchIDListHelper>();
  branchIDListHelper->updateFromRegistry(*preg);
  auto thinnedAssociationsHelper = std::make_shared<edm::ThinnedAssociationsHelper>();
  edm::EventID col(1L, 1L, 1L);
  std::string uuid = edm::createGlobalIdentifier();
  edm::Timestamp fakeTime;
  auto pc = edmtest::makeDummyProcessConfiguration("PROD");
  std::shared_ptr<edm::ProductRegistry const> pregc(preg.release());
  auto rp =
      std::make_shared<edm::RunPrincipal>(pregc, edm::productResolversFactory::makePrimary, pc, &historyAppender_, 0);
  rp->setAux(edm::RunAuxiliary(col.run(), fakeTime, fakeTime));
  edm::LuminosityBlockAuxiliary lumiAux(rp->run(), 1, fakeTime, fakeTime);
  auto lbp = std::make_shared<edm::LuminosityBlockPrincipal>(
      pregc, edm::productResolversFactory::makePrimary, pc, &historyAppender_, 0);
  lbp->setAux(lumiAux);
  lbp->setRunPrincipal(rp);
  edm::EventAuxiliary eventAux(col, uuid, fakeTime, true);
  edm::EventPrincipal ep(pregc,
                         edm::productResolversFactory::makePrimary,
                         branchIDListHelper,
                         thinnedAssociationsHelper,
                         pc,
                         &historyAppender_,
                         edm::StreamID::invalidStreamID());
  ep.fillEventPrincipal(eventAux, nullptr);
  ep.setLuminosityBlockPrincipal(lbp.get());
  try {
    edm::ParameterSet pset;
    pset.registerIt();
    auto processConfiguration = std::make_shared<edm::ProcessConfiguration>();
    edm::ModuleDescription modDesc(
        pset.id(), "Blah", "blahs", processConfiguration.get(), edm::ModuleDescription::getUniqueID());
    edm::Event event(ep, modDesc, nullptr);
    edm::ProducerBase prod;
    event.setProducer(&prod, nullptr);

    std::string label("this does not exist");
    edm::RefProd<edmtest::DummyProduct> ref = event.getRefBeforePut<edmtest::DummyProduct>(label);
    CPPUNIT_ASSERT("Failed to throw required exception" == 0);
  } catch (edm::Exception& x) {
    // nothing to do
  } catch (...) {
    CPPUNIT_ASSERT("Threw wrong kind of exception" == 0);
  }

  try {
    edm::ParameterSet pset;
    pset.registerIt();
    auto processConfiguration = std::make_shared<edm::ProcessConfiguration>();
    edm::ModuleDescription modDesc(
        pset.id(), "Blah", "blahs", processConfiguration.get(), edm::ModuleDescription::getUniqueID());
    edm::Event event(ep, modDesc, nullptr);
    edm::ProducerBase prod;
    event.setProducer(&prod, nullptr);

    std::string label("this does not exist");
    edm::RefProd<edmtest::DummyProduct> ref =
        event.getRefBeforePut<edmtest::DummyProduct>(edm::EDPutTokenT<edmtest::DummyProduct>{});
    CPPUNIT_ASSERT("Failed to throw required exception" == 0);
  } catch (edm::Exception& x) {
    // nothing to do
  } catch (...) {
    CPPUNIT_ASSERT("Threw wrong kind of exception" == 0);
  }
}

void testEventGetRefBeforePut::getRefTest() {
  std::string processName = "PROD";

  std::string label("fred");
  std::string productInstanceName("Rick");

  edmtest::IntProduct dp;
  edm::TypeWithDict dummytype(typeid(edmtest::IntProduct));
  std::string className = dummytype.friendlyClassName();

  edm::ParameterSet dummyProcessPset;
  dummyProcessPset.registerIt();
  auto processConfiguration = std::make_shared<edm::ProcessConfiguration>();
  processConfiguration->setParameterSetID(dummyProcessPset.id());

  edm::ProductDescription product(
      edm::InEvent, label, processName, dummytype.userClassName(), className, productInstanceName, dummytype);

  product.init();

  auto preg = std::make_unique<edm::SignallingProductRegistryFiller>();
  preg->addProduct(product);
  preg->setFrozen();
  auto branchIDListHelper = std::make_shared<edm::BranchIDListHelper>();
  branchIDListHelper->updateFromRegistry(preg->registry());
  auto thinnedAssociationsHelper = std::make_shared<edm::ThinnedAssociationsHelper>();
  edm::EventID col(1L, 1L, 1L);
  std::string uuid = edm::createGlobalIdentifier();
  edm::Timestamp fakeTime;
  auto pcPtr = edmtest::makeSharedDummyProcessConfiguration(processName);
  edm::ProcessConfiguration& pc = *pcPtr;
  std::shared_ptr<edm::ProductRegistry const> pregc(std::make_shared<edm::ProductRegistry>(preg->moveTo()));
  auto rp =
      std::make_shared<edm::RunPrincipal>(pregc, edm::productResolversFactory::makePrimary, pc, &historyAppender_, 0);
  rp->setAux(edm::RunAuxiliary(col.run(), fakeTime, fakeTime));
  edm::LuminosityBlockAuxiliary lumiAux(rp->run(), 1, fakeTime, fakeTime);
  auto lbp = std::make_shared<edm::LuminosityBlockPrincipal>(
      pregc, edm::productResolversFactory::makePrimary, pc, &historyAppender_, 0);
  lbp->setAux(lumiAux);
  lbp->setRunPrincipal(rp);
  edm::EventAuxiliary eventAux(col, uuid, fakeTime, true);
  edm::EventPrincipal ep(pregc,
                         edm::productResolversFactory::makePrimary,
                         branchIDListHelper,
                         thinnedAssociationsHelper,
                         pc,
                         &historyAppender_,
                         edm::StreamID::invalidStreamID());
  ep.fillEventPrincipal(eventAux, nullptr);
  ep.setLuminosityBlockPrincipal(lbp.get());

  edm::RefProd<edmtest::IntProduct> refToProd;
  try {
    edm::ModuleDescription modDesc("Blah", label, pcPtr.get());

    edm::Event event(ep, modDesc, nullptr);
    TestProducer prod(productInstanceName);
    const_cast<std::vector<edm::ProductResolverIndex>&>(prod.putTokenIndexToProductResolverIndex()).push_back(0);
    event.setProducer(&prod, nullptr);
    auto pr = std::make_unique<edmtest::IntProduct>();
    pr->value = 10;

    refToProd = event.getRefBeforePut<edmtest::IntProduct>(productInstanceName);
    event.put(std::move(pr), productInstanceName);
    event.commit_(std::vector<edm::ProductResolverIndex>());
  } catch (cms::Exception& x) {
    std::cerr << x.explainSelf() << std::endl;
    CPPUNIT_ASSERT("Threw exception unexpectedly" == 0);
  } catch (std::exception& x) {
    std::cerr << x.what() << std::endl;
    CPPUNIT_ASSERT("threw std::exception" == 0);
  } catch (...) {
    std::cerr << "Unknown exception type\n";
    CPPUNIT_ASSERT("Threw exception unexpectedly" == 0);
  }
  CPPUNIT_ASSERT(refToProd->value == 10);
}
