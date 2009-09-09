/*----------------------------------------------------------------------

Test of the EventPrincipal class.

----------------------------------------------------------------------*/  
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeinfo>

#include <cppunit/extensions/HelperMacros.h>
#include "boost/shared_ptr.hpp"

#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/Parentage.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ProductStatus.h"
#include "DataFormats/Provenance/interface/Provenance.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "DataFormats/Common/interface/BasicHandle.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/Framework/interface/Selector.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/GetPassID.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"
#include "FWCore/Utilities/interface/GlobalIdentifier.h"

class test_ep: public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(test_ep);
  CPPUNIT_TEST(failgetbyIdTest);
  CPPUNIT_TEST(failgetbySelectorTest);
  CPPUNIT_TEST(failgetbyLabelTest);
  CPPUNIT_TEST(failgetManyTest);
  CPPUNIT_TEST(failgetbyTypeTest);
  CPPUNIT_TEST(failgetManybyTypeTest);
  CPPUNIT_TEST(failgetbyInvalidIdTest);
  CPPUNIT_TEST(failgetProvenanceTest);
  CPPUNIT_TEST_SUITE_END();
public:
  void setUp();
  void tearDown();
  void failgetbyIdTest();
  void failgetbySelectorTest();
  void failgetbyLabelTest();
  void failgetManyTest();
  void failgetbyTypeTest();
  void failgetManybyTypeTest();
  void failgetbyInvalidIdTest();
  void failgetProvenanceTest();

private:

  boost::shared_ptr<edm::ProcessConfiguration> 
  fake_single_module_process(std::string const& tag,
			     std::string const& processName,
			     edm::ParameterSet const& moduleParams,
			     std::string const& release = edm::getReleaseVersion(),
			     std::string const& pass = edm::getPassID() );
  boost::shared_ptr<edm::BranchDescription>
  fake_single_process_branch(std::string const& tag,
			     std::string const& processName, 
			     std::string const& productInstanceName = std::string() );

  std::map<std::string, boost::shared_ptr<edm::BranchDescription> >    branchDescriptions_;
  std::map<std::string, boost::shared_ptr<edm::ProcessConfiguration> > processConfigurations_;
  
  boost::shared_ptr<edm::ProductRegistry>   pProductRegistry_;
  boost::shared_ptr<edm::EventPrincipal>    pEvent_;
  std::vector<edm::ProductID> contained_products_;
  
  edm::EventID               eventID_;
};

//----------------------------------------------------------------------
// registration of the test so that the runner can find it

CPPUNIT_TEST_SUITE_REGISTRATION(test_ep);


//----------------------------------------------------------------------

boost::shared_ptr<edm::ProcessConfiguration>
test_ep::fake_single_module_process(std::string const& tag,
				    std::string const& processName,
				    edm::ParameterSet const& moduleParams,
				    std::string const& release,
				    std::string const& pass) {
  edm::ParameterSet processParams;
  processParams.addParameter(processName, moduleParams);
  processParams.addParameter<std::string>("@process_name",
					  processName);
  
  processParams.registerIt();
  boost::shared_ptr<edm::ProcessConfiguration> result(
    new edm::ProcessConfiguration(processName, processParams.id(), release, pass));
  processConfigurations_[tag] = result;
  return result;
}

boost::shared_ptr<edm::BranchDescription>
test_ep::fake_single_process_branch(std::string const& tag, 
				    std::string const& processName,
				    std::string const& productInstanceName) {
  std::string moduleLabel = processName + "dummyMod";
  std::string moduleClass("DummyModule");
  edm::TypeID dummyType(typeid(edmtest::DummyProduct));
  std::string productClassName = dummyType.userClassName();
  std::string friendlyProductClassName = dummyType.friendlyClassName();
  edm::ParameterSet modParams;
  modParams.addParameter<std::string>("@module_type", moduleClass);
  modParams.addParameter<std::string>("@module_label", moduleLabel);
  modParams.registerIt();
  boost::shared_ptr<edm::ProcessConfiguration> process(fake_single_module_process(tag, processName, modParams));
  boost::shared_ptr<edm::ProcessConfiguration> processX(new edm::ProcessConfiguration(*process));
  edm::ModuleDescription mod(modParams.id(), moduleClass, moduleLabel, processX);

  boost::shared_ptr<edm::BranchDescription> result(
    new edm::BranchDescription(edm::InEvent, 
			       moduleLabel, 
			       processName,
			       productClassName,
			       friendlyProductClassName,
			       productInstanceName,
			       mod));
  branchDescriptions_[tag] = result;
  return result;
}

void test_ep::setUp() {
  edm::BranchIDListHelper::clearRegistries();

  // Making a functional EventPrincipal is not trivial, so we do it
  // all here.
  eventID_ = edm::EventID(101, 20);

  // We can only insert products registered in the ProductRegistry.
  pProductRegistry_.reset(new edm::ProductRegistry);
  pProductRegistry_->addProduct(*fake_single_process_branch("hlt",  "HLT"));
  pProductRegistry_->addProduct(*fake_single_process_branch("prod", "PROD"));
  pProductRegistry_->addProduct(*fake_single_process_branch("test", "TEST"));
  pProductRegistry_->addProduct(*fake_single_process_branch("user", "USER"));
  pProductRegistry_->addProduct(*fake_single_process_branch("rick", "USER2", "rick"));
  pProductRegistry_->setFrozen();
  edm::BranchIDListHelper::updateRegistries(*pProductRegistry_);
 
  // Put products we'll look for into the EventPrincipal.
  {
    typedef edmtest::DummyProduct PRODUCT_TYPE;
    typedef edm::Wrapper<PRODUCT_TYPE> WDP;
    std::auto_ptr<edm::EDProduct> product(new WDP(std::auto_ptr<PRODUCT_TYPE>(new PRODUCT_TYPE)));

    std::string tag("rick");
    assert(branchDescriptions_[tag]);
    edm::BranchDescription branch = *branchDescriptions_[tag];

    branch.init();

    edm::ProductRegistry::ProductList const& pl = pProductRegistry_->productList();
    edm::BranchKey const bk(branch);
    edm::ProductRegistry::ProductList::const_iterator it = pl.find(bk);

    edm::ConstBranchDescription const branchFromRegistry(it->second);

    boost::shared_ptr<edm::Parentage> entryDescriptionPtr(new edm::Parentage);
    std::auto_ptr<edm::ProductProvenance> branchEntryInfoPtr(
      new edm::ProductProvenance(branchFromRegistry.branchID(),
                               edm::productstatus::present(),
                               entryDescriptionPtr));

    boost::shared_ptr<edm::ProcessConfiguration> process(processConfigurations_[tag]);
    assert(process);
    std::string uuid = edm::createGlobalIdentifier();
    edm::Timestamp now(1234567UL);
    boost::shared_ptr<edm::RunAuxiliary> runAux(new edm::RunAuxiliary(eventID_.run(), now, now));
    boost::shared_ptr<edm::RunPrincipal> rp(new edm::RunPrincipal(runAux, pProductRegistry_, *process));
    boost::shared_ptr<edm::LuminosityBlockAuxiliary> lumiAux(new edm::LuminosityBlockAuxiliary(rp->run(), 1, now, now));
    boost::shared_ptr<edm::LuminosityBlockPrincipal>lbp(new edm::LuminosityBlockPrincipal(lumiAux, pProductRegistry_, *process, rp));
    std::auto_ptr<edm::EventAuxiliary> eventAux(new edm::EventAuxiliary(eventID_, uuid, now, lbp->luminosityBlock(), true));
    pEvent_.reset(new edm::EventPrincipal(pProductRegistry_, *process));
    pEvent_->fillEventPrincipal(eventAux, lbp);
    pEvent_->put(branchFromRegistry, product, branchEntryInfoPtr);
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

void test_ep::failgetbySelectorTest() {
  // We don't put ProductIDs into the EventPrincipal,
  // so that's a type sure not to match any product.
  edm::ProductID dummy;
  edm::TypeID tid(dummy);

  edm::ProcessNameSelector pnsel("PROD");
  edm::BasicHandle h(pEvent_->getBySelector(tid, pnsel));
  CPPUNIT_ASSERT(h.failedToGet());
}

void test_ep::failgetbyLabelTest() {
  // We don't put ProductIDs into the EventPrincipal,
  // so that's a type sure not to match any product.
  edm::ProductID dummy;
  edm::TypeID tid(dummy);

  std::string label("this does not exist");

  size_t cachedOffset = 0;
  int fillCount = -1;
  edm::BasicHandle h(pEvent_->getByLabel(tid, label, std::string(), std::string(), cachedOffset, fillCount));
  CPPUNIT_ASSERT(h.failedToGet());
}

void test_ep::failgetManyTest() {
  // We don't put ProductIDs into the EventPrincipal,
  // so that's a type sure not to match any product.
  edm::ProductID dummy;
  edm::TypeID tid(dummy);

  edm::ProcessNameSelector sel("PROD");
  std::vector<edm::BasicHandle > handles;
  pEvent_->getMany(tid, sel, handles);
  CPPUNIT_ASSERT(handles.empty());
}

void test_ep::failgetbyTypeTest() {
  edm::ProductID dummy;
  edm::TypeID tid(dummy);
  edm::BasicHandle h(pEvent_->getByType(tid));
  CPPUNIT_ASSERT(h.failedToGet());
}

void test_ep::failgetManybyTypeTest() {
  // We don't put ProductIDs into the EventPrincipal,
  // so that's a type sure not to match any product.
  edm::ProductID dummy;
  edm::TypeID tid(dummy);
  std::vector<edm::BasicHandle > handles;

  
  pEvent_->getManyByType(tid, handles);
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

