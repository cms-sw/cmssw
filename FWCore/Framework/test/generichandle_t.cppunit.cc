/*----------------------------------------------------------------------

Test of GenericHandle class.

----------------------------------------------------------------------*/
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/GenericHandle.h"
#include "FWCore/Framework/interface/HistoryAppender.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/GetPassID.h"
#include "FWCore/Utilities/interface/GlobalIdentifier.h"
#include "FWCore/Reflection/interface/TypeWithDict.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"

#include "cppunit/extensions/HelperMacros.h"

#include <iostream>
#include <memory>
#include <string>

// This is a gross hack, to allow us to test the event
namespace edm {
  class ProducerBase {
  public:
    static void commitEvent(Event& e) { e.commit_(std::vector<ProductResolverIndex>()); }
  };
}  // namespace edm

class testGenericHandle : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testGenericHandle);
  CPPUNIT_TEST(failgetbyLabelTest);
  CPPUNIT_TEST(getbyLabelTest);
  CPPUNIT_TEST(failWrongType);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}
  void failgetbyLabelTest();
  void failWrongType();
  void getbyLabelTest();

  edm::HistoryAppender historyAppender_;
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testGenericHandle);

void testGenericHandle::failWrongType() {
  try {
    //intentionally misspelled type
    edm::GenericHandle h("edmtest::DmmyProduct");
    CPPUNIT_ASSERT("Failed to throw" == nullptr);
  } catch (cms::Exception& x) {
    // nothing to do
  } catch (...) {
    CPPUNIT_ASSERT("Threw wrong kind of exception" == nullptr);
  }
}
void testGenericHandle::failgetbyLabelTest() {
  edm::EventID id = edm::EventID::firstValidEvent();
  edm::Timestamp time;
  std::string uuid = edm::createGlobalIdentifier();
  edm::ProcessConfiguration pc("PROD", edm::ParameterSetID(), edm::getReleaseVersion(), edm::getPassID());
  auto preg = std::make_shared<edm::ProductRegistry>();
  preg->setFrozen();
  auto runAux = std::make_shared<edm::RunAuxiliary>(id.run(), time, time);
  auto rp = std::make_shared<edm::RunPrincipal>(runAux, preg, pc, &historyAppender_, 0);
  auto lbp = std::make_shared<edm::LuminosityBlockPrincipal>(preg, pc, &historyAppender_, 0);
  lbp->setAux(edm::LuminosityBlockAuxiliary(rp->run(), 1, time, time));
  lbp->setRunPrincipal(rp);
  auto branchIDListHelper = std::make_shared<edm::BranchIDListHelper>();
  branchIDListHelper->updateFromRegistry(*preg);
  auto thinnedAssociationsHelper = std::make_shared<edm::ThinnedAssociationsHelper>();
  edm::EventAuxiliary eventAux(id, uuid, time, true);
  edm::EventPrincipal ep(
      preg, branchIDListHelper, thinnedAssociationsHelper, pc, &historyAppender_, edm::StreamID::invalidStreamID());
  ep.fillEventPrincipal(eventAux, nullptr);
  ep.setLuminosityBlockPrincipal(lbp.get());
  edm::GenericHandle h("edmtest::DummyProduct");
  bool didThrow = true;
  try {
    edm::ParameterSet pset;
    pset.registerIt();
    edm::ModuleDescription modDesc(pset.id(), "Blah", "blahs");
    edm::Event event(ep, modDesc, nullptr);

    std::string label("this does not exist");
    event.getByLabel(label, h);
    *h;
    didThrow = false;
  } catch (cms::Exception& x) {
    // nothing to do
  } catch (std::exception& x) {
    std::cout << "caught std exception " << x.what() << std::endl;
    CPPUNIT_ASSERT("Threw std::exception!" == nullptr);
  } catch (...) {
    CPPUNIT_ASSERT("Threw wrong kind of exception" == nullptr);
  }
  if (!didThrow) {
    CPPUNIT_ASSERT("Failed to throw required exception" == nullptr);
  }
}

void testGenericHandle::getbyLabelTest() {
  std::string processName = "PROD";

  typedef edmtest::DummyProduct DP;
  typedef edm::Wrapper<DP> WDP;

  auto pr = std::make_unique<DP>();
  std::unique_ptr<edm::WrapperBase> pprod = std::make_unique<WDP>(std::move(pr));
  std::string label("fred");
  std::string productInstanceName("Rick");

  edm::TypeWithDict dummytype(typeid(edmtest::DummyProduct));
  std::string className = dummytype.friendlyClassName();

  edm::ParameterSet dummyProcessPset;
  dummyProcessPset.registerIt();

  edm::ParameterSet pset;
  pset.registerIt();

  edm::BranchDescription product(edm::InEvent,
                                 label,
                                 processName,
                                 dummytype.userClassName(),
                                 className,
                                 productInstanceName,
                                 "",
                                 pset.id(),
                                 dummytype);

  product.init();

  auto preg = std::make_unique<edm::ProductRegistry>();
  preg->addProduct(product);
  preg->setFrozen();
  auto branchIDListHelper = std::make_shared<edm::BranchIDListHelper>();
  branchIDListHelper->updateFromRegistry(*preg);
  auto thinnedAssociationsHelper = std::make_shared<edm::ThinnedAssociationsHelper>();

  edm::ProductRegistry::ProductList const& pl = preg->productList();
  edm::BranchKey const bk(product);
  edm::ProductRegistry::ProductList::const_iterator it = pl.find(bk);

  edm::EventID col(1L, 1L, 1L);
  edm::Timestamp fakeTime;
  std::string uuid = edm::createGlobalIdentifier();
  edm::ProcessConfiguration pc("PROD", dummyProcessPset.id(), edm::getReleaseVersion(), edm::getPassID());
  std::shared_ptr<edm::ProductRegistry const> pregc(preg.release());
  auto runAux = std::make_shared<edm::RunAuxiliary>(col.run(), fakeTime, fakeTime);
  auto rp = std::make_shared<edm::RunPrincipal>(runAux, pregc, pc, &historyAppender_, 0);
  auto lbp = std::make_shared<edm::LuminosityBlockPrincipal>(pregc, pc, &historyAppender_, 0);
  lbp->setAux(edm::LuminosityBlockAuxiliary(rp->run(), 1, fakeTime, fakeTime));
  lbp->setRunPrincipal(rp);
  edm::EventAuxiliary eventAux(col, uuid, fakeTime, true);
  edm::EventPrincipal ep(
      pregc, branchIDListHelper, thinnedAssociationsHelper, pc, &historyAppender_, edm::StreamID::invalidStreamID());
  ep.fillEventPrincipal(eventAux, nullptr);
  ep.setLuminosityBlockPrincipal(lbp.get());
  edm::BranchDescription const& branchFromRegistry = it->second;
  std::vector<edm::BranchID> const ids;
  edm::ProductProvenance prov(branchFromRegistry.branchID(), ids);
  edm::BranchDescription const desc(branchFromRegistry);
  ep.put(desc, std::move(pprod), prov);

  edm::GenericHandle h("edmtest::DummyProduct");
  try {
    auto processConfiguration = std::make_shared<edm::ProcessConfiguration>();
    processConfiguration->setParameterSetID(dummyProcessPset.id());

    edm::ModuleDescription modDesc(
        pset.id(), "Blah", "blahs", processConfiguration.get(), edm::ModuleDescription::getUniqueID());
    edm::Event event(ep, modDesc, nullptr);

    event.getByLabel(label, productInstanceName, h);
  } catch (cms::Exception& x) {
    std::cerr << x.explainSelf() << std::endl;
    CPPUNIT_ASSERT("Threw cms::Exception unexpectedly" == nullptr);
  } catch (std::exception& x) {
    std::cerr << x.what() << std::endl;
    CPPUNIT_ASSERT("threw std::exception" == nullptr);
  } catch (...) {
    std::cerr << "Unknown exception type\n";
    CPPUNIT_ASSERT("Threw exception unexpectedly" == nullptr);
  }
  CPPUNIT_ASSERT(h.isValid());
  CPPUNIT_ASSERT(h.provenance()->moduleLabel() == label);
}
