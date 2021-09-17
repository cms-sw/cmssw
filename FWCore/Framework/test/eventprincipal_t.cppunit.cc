/*----------------------------------------------------------------------

Test of the EventPrincipal class.

----------------------------------------------------------------------*/
#include "DataFormats/Common/interface/BasicHandle.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/Parentage.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/HistoryAppender.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/GetPassID.h"
#include "FWCore/Utilities/interface/GlobalIdentifier.h"
#include "FWCore/Utilities/interface/ProductKindOfType.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Reflection/interface/TypeWithDict.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"

#include "cppunit/extensions/HelperMacros.h"

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeinfo>

class test_ep : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(test_ep);
  CPPUNIT_TEST(failgetbyIdTest);
  CPPUNIT_TEST(failgetbyLabelTest);
  CPPUNIT_TEST(failgetManybyTypeTest);
  CPPUNIT_TEST(failgetbyInvalidIdTest);
  CPPUNIT_TEST(failgetProvenanceTest);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp();
  void tearDown();
  void failgetbyIdTest();
  void failgetbyLabelTest();
  void failgetManybyTypeTest();
  void failgetbyInvalidIdTest();
  void failgetProvenanceTest();

private:
  std::shared_ptr<edm::ProcessConfiguration> fake_single_module_process(
      std::string const& tag,
      std::string const& processName,
      edm::ParameterSet const& moduleParams,
      std::string const& release = edm::getReleaseVersion(),
      std::string const& pass = edm::getPassID());
  std::shared_ptr<edm::BranchDescription> fake_single_process_branch(
      std::string const& tag, std::string const& processName, std::string const& productInstanceName = std::string());

  std::map<std::string, std::shared_ptr<edm::BranchDescription> > branchDescriptions_;
  std::map<std::string, std::shared_ptr<edm::ProcessConfiguration> > processConfigurations_;

  std::shared_ptr<edm::ProductRegistry> pProductRegistry_;
  std::shared_ptr<edm::LuminosityBlockPrincipal> lbp_;
  std::shared_ptr<edm::EventPrincipal> pEvent_;

  edm::EventID eventID_;

  edm::HistoryAppender historyAppender_;
};

//----------------------------------------------------------------------
// registration of the test so that the runner can find it

CPPUNIT_TEST_SUITE_REGISTRATION(test_ep);

//----------------------------------------------------------------------

std::shared_ptr<edm::ProcessConfiguration> test_ep::fake_single_module_process(std::string const& tag,
                                                                               std::string const& processName,
                                                                               edm::ParameterSet const& moduleParams,
                                                                               std::string const& release,
                                                                               std::string const& pass) {
  edm::ParameterSet processParams;
  processParams.addParameter(processName, moduleParams);
  processParams.addParameter<std::string>("@process_name", processName);

  processParams.registerIt();
  auto result = std::make_shared<edm::ProcessConfiguration>(processName, processParams.id(), release, pass);
  processConfigurations_[tag] = result;
  return result;
}

std::shared_ptr<edm::BranchDescription> test_ep::fake_single_process_branch(std::string const& tag,
                                                                            std::string const& processName,
                                                                            std::string const& productInstanceName) {
  std::string moduleLabel = processName + "dummyMod";
  std::string moduleClass("DummyModule");
  edm::TypeWithDict dummyType(typeid(edmtest::DummyProduct));
  std::string productClassName = dummyType.userClassName();
  std::string friendlyProductClassName = dummyType.friendlyClassName();
  edm::ParameterSet modParams;
  modParams.addParameter<std::string>("@module_type", moduleClass);
  modParams.addParameter<std::string>("@module_label", moduleLabel);
  modParams.registerIt();
  std::shared_ptr<edm::ProcessConfiguration> process(fake_single_module_process(tag, processName, modParams));

  auto result = std::make_shared<edm::BranchDescription>(edm::InEvent,
                                                         moduleLabel,
                                                         processName,
                                                         productClassName,
                                                         friendlyProductClassName,
                                                         productInstanceName,
                                                         moduleClass,
                                                         modParams.id(),
                                                         dummyType);
  branchDescriptions_[tag] = result;
  return result;
}

void test_ep::setUp() {
  // Making a functional EventPrincipal is not trivial, so we do it
  // all here.
  eventID_ = edm::EventID(101, 1, 20);

  // We can only insert products registered in the ProductRegistry.
  pProductRegistry_.reset(new edm::ProductRegistry);
  pProductRegistry_->addProduct(*fake_single_process_branch("hlt", "HLT"));
  pProductRegistry_->addProduct(*fake_single_process_branch("prod", "PROD"));
  pProductRegistry_->addProduct(*fake_single_process_branch("test", "TEST"));
  pProductRegistry_->addProduct(*fake_single_process_branch("user", "USER"));
  pProductRegistry_->addProduct(*fake_single_process_branch("rick", "USER2", "rick"));
  pProductRegistry_->setFrozen();
  auto branchIDListHelper = std::make_shared<edm::BranchIDListHelper>();
  branchIDListHelper->updateFromRegistry(*pProductRegistry_);
  auto thinnedAssociationsHelper = std::make_shared<edm::ThinnedAssociationsHelper>();

  // Put products we'll look for into the EventPrincipal.
  {
    typedef edmtest::DummyProduct PRODUCT_TYPE;
    typedef edm::Wrapper<PRODUCT_TYPE> WDP;

    std::unique_ptr<edm::WrapperBase> product = std::make_unique<WDP>(std::make_unique<PRODUCT_TYPE>());

    std::string tag("rick");
    assert(branchDescriptions_[tag]);
    edm::BranchDescription branch = *branchDescriptions_[tag];

    branch.init();

    edm::ProductRegistry::ProductList const& pl = pProductRegistry_->productList();
    edm::BranchKey const bk(branch);
    edm::ProductRegistry::ProductList::const_iterator it = pl.find(bk);

    edm::BranchDescription const branchFromRegistry(it->second);

    std::vector<edm::BranchID> const ids;
    edm::ProductProvenance prov(branchFromRegistry.branchID(), ids);

    std::shared_ptr<edm::ProcessConfiguration> process(processConfigurations_[tag]);
    assert(process);
    std::string uuid = edm::createGlobalIdentifier();
    edm::Timestamp now(1234567UL);
    auto runAux = std::make_shared<edm::RunAuxiliary>(eventID_.run(), now, now);
    auto rp = std::make_shared<edm::RunPrincipal>(runAux, pProductRegistry_, *process, &historyAppender_, 0);
    edm::LuminosityBlockAuxiliary lumiAux(rp->run(), 1, now, now);
    lbp_ = std::make_shared<edm::LuminosityBlockPrincipal>(pProductRegistry_, *process, &historyAppender_, 0);
    lbp_->setAux(lumiAux);
    lbp_->setRunPrincipal(rp);
    edm::EventAuxiliary eventAux(eventID_, uuid, now, true);
    pEvent_.reset(new edm::EventPrincipal(pProductRegistry_,
                                          branchIDListHelper,
                                          thinnedAssociationsHelper,
                                          *process,
                                          &historyAppender_,
                                          edm::StreamID::invalidStreamID()));
    pEvent_->fillEventPrincipal(eventAux, nullptr);
    pEvent_->setLuminosityBlockPrincipal(lbp_.get());
    pEvent_->put(branchFromRegistry, std::move(product), prov);
  }
  CPPUNIT_ASSERT(pEvent_->size() == 1);
}

template <class MAP>
void clear_map(MAP& m) {
  for (typename MAP::iterator i = m.begin(), e = m.end(); i != e; ++i)
    i->second.reset();
}

void test_ep::tearDown() {
  clear_map(branchDescriptions_);
  clear_map(processConfigurations_);

  pEvent_.reset();

  pProductRegistry_.reset();
}

//----------------------------------------------------------------------
// Test functions
//----------------------------------------------------------------------

void test_ep::failgetbyIdTest() {
  edm::ProductID invalid;
  CPPUNIT_ASSERT_THROW(pEvent_->getByProductID(invalid), edm::Exception);

  edm::ProductID notpresent(0, 10000);
  edm::BasicHandle h(pEvent_->getByProductID(notpresent));
  CPPUNIT_ASSERT(h.failedToGet());
}

void test_ep::failgetbyLabelTest() {
  edmtest::IntProduct dummy;
  edm::TypeID tid(dummy);

  std::string label("this does not exist");

  edm::BasicHandle h(
      pEvent_->getByLabel(edm::PRODUCT_TYPE, tid, label, std::string(), std::string(), nullptr, nullptr, nullptr));
  CPPUNIT_ASSERT(h.failedToGet());
}

void test_ep::failgetManybyTypeTest() {
  edmtest::IntProduct dummy;
  edm::TypeID tid(dummy);
  std::vector<edm::BasicHandle> handles;

  pEvent_->getManyByType(tid, handles, nullptr, nullptr, nullptr);
  CPPUNIT_ASSERT(handles.empty());
}

void test_ep::failgetbyInvalidIdTest() {
  //put_a_dummy_product("HLT");
  //put_a_product<edmtest::DummyProduct>(pProdConfig_, label);

  edm::ProductID id;
  CPPUNIT_ASSERT_THROW(pEvent_->getByProductID(id), edm::Exception);
}

void test_ep::failgetProvenanceTest() {
  edm::BranchID id;
  CPPUNIT_ASSERT_THROW(pEvent_->getProvenance(id), edm::Exception);
}
