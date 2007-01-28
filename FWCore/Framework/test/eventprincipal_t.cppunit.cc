/*----------------------------------------------------------------------

Test of the EventPrincipal class.

$Id: eventprincipal_t.cppunit.cc,v 1.36 2007/01/12 21:07:59 wmtan Exp $

----------------------------------------------------------------------*/  
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeinfo>

#include <cppunit/extensions/HelperMacros.h>

#include "DataFormats/Common/interface/BranchDescription.h"
#include "DataFormats/Common/interface/ModuleDescription.h"
#include "DataFormats/Common/interface/ParameterSetID.h"
#include "DataFormats/Common/interface/ProcessConfiguration.h"
#include "DataFormats/Common/interface/ProductID.h"
#include "DataFormats/Common/interface/ProductRegistry.h"
#include "DataFormats/Common/interface/Timestamp.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "DataFormats/Common/interface/BasicHandle.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Selector.h"
#include "FWCore/Framework/interface/TypeID.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/GetPassID.h"
#include "FWCore/Utilities/interface/GetReleaseVersion.h"
#include "FWCore/Utilities/interface/PretendToUse.h"
#include "FWCore/Utilities/interface/value_ptr.h"

typedef edm::BasicHandle handle;

class testeventprincipal: public CppUnit::TestFixture 
{
  CPPUNIT_TEST_SUITE(testeventprincipal);
  CPPUNIT_TEST(failgetbyIdTest);
  CPPUNIT_TEST(failgetbySelectorTest);
  CPPUNIT_TEST(failgetbyLabelTest);
  CPPUNIT_TEST(failgetManyTest);
  CPPUNIT_TEST(failgetbyTypeTest);
  CPPUNIT_TEST(failgetManybyTypeTest);
  CPPUNIT_TEST(failgetbyInvalidIdTest);
  CPPUNIT_TEST(failgetProvenanceTest);
  //   CPPUNIT_TEST(getbyIdTest);
  //   CPPUNIT_TEST(getbySelectorTest);
  //   CPPUNIT_TEST(getbyLabelTest);
  //   CPPUNIT_TEST(getbyTypeTest);
  //   CPPUNIT_TEST(getProvenanceTest);
  //   CPPUNIT_TEST(getAllProvenanceTest);
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

  //   void getAllProvenanceTest();
  //   void getbyIdTest();
  //   void getbySelectorTest();
  //   void processNameSelectorTest();
  //   void getbyLabelTest();
  //   void getbyTypeTest();
  //   void getProvenanceTest();

 private:

  edm::ProcessConfiguration* 
  fake_single_module_process(std::string const& tag,
			     std::string const& processName,
			     edm::ParameterSet const& moduleParams,
			     std::string const& release = edm::getReleaseVersion(),
			     std::string const& pass = edm::getPassID() );
  edm::BranchDescription*
  fake_single_process_branch(std::string const& tag,
			     std::string const& processName, 
			     std::string const& productInstanceName = std::string() );
  // Put a DummyProduct into the EventPrincipal, recorded as having come from
  // the module and process identified by the 'tag'.
  //void put_a_dummy_product(std::string const& tag);
  template <class PRODUCT_TYPE>
  void 
  put_a_product(edm::ProcessConfiguration* config,
		std::string const& moduleLabel,
		std::string const& productInstanceName = std::string() );
  
  //   edm::ProcessConfiguration* pHltConfig_;
  //   edm::ProcessConfiguration* pProdConfig_;  
  //   edm::ProcessConfiguration* pTestConfig_;
  //   edm::ProcessConfiguration* pUserConfig_;
  
  std::map<std::string, edm::BranchDescription*>    branchDescriptions_;
  std::map<std::string, edm::ProcessConfiguration*> processConfigurations_;
  
  edm::ProductRegistry*      pProductRegistry_;
  edm::EventPrincipal*       pEvent_;
  
  edm::EventID               eventID_;
};

edm::ProcessConfiguration*
testeventprincipal::fake_single_module_process(std::string const& tag,
					       std::string const& processName,
					       edm::ParameterSet const& moduleParams,
					       std::string const& release,
					       std::string const& pass)
{
  edm::ParameterSet processParams;
  processParams.addParameter(processName, moduleParams);
  processParams.addParameter<std::string>("@process_name",
					  processName);
  
  edm::ProcessConfiguration* result = 
    new edm::ProcessConfiguration(processName, processParams.id(), release, pass);
  processConfigurations_[tag] = result;
  return result;
}

edm::BranchDescription*
testeventprincipal::fake_single_process_branch(std::string const& tag, 
					       std::string const& processName,
					       std::string const& productInstanceName)
{
  edm::ModuleDescription mod;
  std::string moduleLabel = processName + "dummyMod";
  std::string moduleClass("DummyModule");
  edm::TypeID dummyType(typeid(edmtest::DummyProduct));
  std::string productClassName = dummyType.userClassName();
  std::string friendlyProductClassName = dummyType.friendlyClassName();
  edm::ParameterSet modParams;
  modParams.addParameter<std::string>("@module_type", moduleClass);
  modParams.addParameter<std::string>("@module_label", moduleLabel);
  mod.parameterSetID_ = modParams.id();
  mod.moduleName_ = moduleClass;
  mod.moduleLabel_ = moduleLabel;
  edm::ProcessConfiguration* process = 
    fake_single_module_process(tag, processName, modParams);
  mod.processConfiguration_ = *process;

  edm::BranchDescription* result = 
    new edm::BranchDescription(edm::InEvent, 
			       moduleLabel, 
			       processName,
			       productClassName,
			       friendlyProductClassName,
			       productInstanceName,
			       mod);
  branchDescriptions_[tag] = result;
  return result;
}

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testeventprincipal);

void testeventprincipal::setUp()
{
  // Making a functional EventPrincipal is not trivial, so we do it
  // all here.
  eventID_ = edm::EventID(101, 20, false);

  pProductRegistry_ = new edm::ProductRegistry;

  pProductRegistry_->addProduct(*fake_single_process_branch("hlt",  "HLT"));
  pProductRegistry_->addProduct(*fake_single_process_branch("prod", "PROD"));
  pProductRegistry_->addProduct(*fake_single_process_branch("test", "TEST"));
  pProductRegistry_->addProduct(*fake_single_process_branch("user", "USER"));
  pProductRegistry_->addProduct(*fake_single_process_branch("rick", "USER2", "rick"));
  pProductRegistry_->setProductIDs();
 

  // Put products we'll look for into the EventPrincipal.
  {
    typedef edmtest::DummyProduct PRODUCT_TYPE;
    typedef edm::Wrapper<PRODUCT_TYPE> WDP;
    std::auto_ptr<edm::EDProduct>  product(new WDP(std::auto_ptr<PRODUCT_TYPE>(new PRODUCT_TYPE)));

    std::string tag("rick");
    assert(branchDescriptions_[tag]);
    edm::BranchDescription branch = *branchDescriptions_[tag];

    branch.init();

    edm::ProductRegistry::ProductList const& pl = pProductRegistry_->productList();
    edm::BranchKey const bk(branch);
    edm::ProductRegistry::ProductList::const_iterator it = pl.find(bk);
    branch.productID_ = it->second.productID_;

    std::auto_ptr<edm::Provenance> provenance(new edm::Provenance(branch, edm::BranchEntryDescription::Success));

    edm::ProcessConfiguration* process = processConfigurations_[tag];
    assert(process);
    edm::Timestamp now(1234567UL);
    pEvent_  = new edm::EventPrincipal(eventID_, now, *pProductRegistry_, *process);
    pEvent_->put(product, provenance);
  }
  
}

template <class MAP>
void clear_map(MAP& m)
{
  for (typename MAP::iterator i = m.begin(), e = m.end(); i != e; ++i)
    delete i->second;
}

void testeventprincipal::tearDown()
{

  clear_map(branchDescriptions_);
  clear_map(processConfigurations_);  

  delete pEvent_;
  pEvent_ = 0;

  delete pProductRegistry_;
  pProductRegistry_ = 0;
}

#if 0
void 
testeventprincipal::put_a_dummy_product(std::string const& tag)
{
  edm::ProcessConfiguration* config = processConfigurations_[tag];
  assert(config);

  edm::BranchDescription* branch = branchDescriptions_[tag];
  assert(branch);

  put_a_product<edmtest::DummyProduct>(config,
 				       branch->moduleLabel_,
 				       branch->productInstanceName_);
}
#endif

template <class PRODUCT_TYPE>
void testeventprincipal::put_a_product(edm::ProcessConfiguration* config,
				       std::string const& moduleLabel,
				       std::string const& productInstanceName)
{
  typedef edm::Wrapper<PRODUCT_TYPE> WDP;
  std::auto_ptr<edm::EDProduct>  product(new WDP(std::auto_ptr<PRODUCT_TYPE>(new PRODUCT_TYPE)));
  std::auto_ptr<edm::Provenance> provenance(new edm::Provenance);

  PRODUCT_TYPE dummyProduct;
  edm::TypeID dummytype(dummyProduct);
  std::string className = dummytype.friendlyClassName();

  provenance->product.fullClassName_       = dummytype.userClassName();
  provenance->product.friendlyClassName_   = className;
  provenance->product.moduleLabel_         = moduleLabel;
  provenance->product.processName_         = config->processName();
  provenance->product.productInstanceName_ = productInstanceName;
  provenance->product.init();

  //   pProductRegistry_->addProduct(provenance->product);
  //   pProductRegistry_->setProductIDs();

  edm::ProductRegistry::ProductList const& pl = pProductRegistry_->productList();
  edm::BranchKey const bk(provenance->product);
  edm::ProductRegistry::ProductList::const_iterator it = pl.find(bk);
  provenance->product.productID_ = it->second.productID_;

  //   edm::EventID col(1L);
  //   edm::Timestamp fakeTime;
  //   edm::EventPrincipal ep(col, fakeTime, *pProductRegistry_, *pProdConfig_);
  pEvent_->put(product, provenance);
}

//----------------------------------------------------------------------
// Test functions
//----------------------------------------------------------------------

void testeventprincipal::failgetbyIdTest() 
{
  edm::ProductID invalid;
  CPPUNIT_ASSERT_THROW(pEvent_->get(invalid), edm::Exception);

  edm::ProductID notpresent(10000000);
  CPPUNIT_ASSERT_THROW(pEvent_->get(notpresent), edm::Exception);
}

void testeventprincipal::failgetbySelectorTest()
{
  // We don't put EventPrincipals into the EventPrincipal,
  // so that's a type sure not to match any product.
  edm::TypeID tid(*pEvent_); 

  edm::ProcessNameSelector pnsel("PROD");
  CPPUNIT_ASSERT_THROW(pEvent_->getBySelector(tid, pnsel), edm::Exception);
}

void testeventprincipal::failgetbyLabelTest() 
{
  // We don't put EventPrincipals into the EventPrincipal,
  // so that's a type sure not to match any product.
  edm::TypeID tid(*pEvent_);

  std::string label("this does not exist");

  CPPUNIT_ASSERT_THROW(pEvent_->getByLabel(tid, label, std::string()),
		       edm::Exception);
}

void testeventprincipal::failgetManyTest() 
{
  // We don't put EventPrincipals into the EventPrincipal,
  // so that's a type sure not to match any product.
  edm::TypeID tid(*pEvent_);

  edm::ProcessNameSelector sel("PROD");
  std::vector<handle> handles;
  CPPUNIT_ASSERT_THROW(pEvent_->getMany(tid, sel, handles),
		       edm::Exception);
}

void testeventprincipal::failgetbyTypeTest() 
{
  edm::TypeID tid(*pEvent_);
  CPPUNIT_ASSERT_THROW(pEvent_->getByType(tid), edm::Exception);
}

void testeventprincipal::failgetManybyTypeTest() 
{
  // We don't put EventPrincipals into the EventPrincipal,
  // so that's a type sure not to match any product.
  edm::TypeID tid(*pEvent_);
  std::vector<handle> handles;

  // TODO: Why does this throw? The design was for getManyByType NOT
  // to throw if no matches were found -- it can just return an empty
  // collection!
  CPPUNIT_ASSERT_THROW(pEvent_->getManyByType(tid, handles),
		       edm::Exception);
}

void testeventprincipal::failgetbyInvalidIdTest() 
{
  //put_a_dummy_product("HLT");
  //put_a_product<edmtest::DummyProduct>(pProdConfig_, label);

  edm::ProductID id;
  CPPUNIT_ASSERT_THROW(pEvent_->get(id), edm::Exception);
}

void testeventprincipal::failgetProvenanceTest() 
{
  edm::ProductID id;
  CPPUNIT_ASSERT_THROW(pEvent_->getProvenance(id), edm::Exception);
}

#if 0
void testeventprincipal::getbyIdTest() 
{
  //   std::string label("modulename");
  //   put_a_product<edmtest::DummyProduct>(pProdConfig_, label);
  put_a_dummy_product("PROD");
  edm::ProductID id(1);
  handle h = pEvent_->get(id);
  CPPUNIT_ASSERT(h.isValid());
  CPPUNIT_ASSERT(h.id() == id);
}

void testeventprincipal::getbyLabelTest() 
{
  std::string label("fred");
  std::string productInstanceName("Rick");
  put_a_product<edmtest::DummyProduct>(pProdConfig_, label, productInstanceName);
  
  edmtest::DummyProduct example;
  edm::TypeID tid(example);

  handle h = pEvent_->getByLabel(tid, label, productInstanceName);
  CPPUNIT_ASSERT(h.isValid());
  CPPUNIT_ASSERT(h.provenance()->moduleLabel() == label);
  {
    handle h = pEvent_->getByLabel(tid, 
				   label, 
				   productInstanceName, 
				   pProdConfig_->processName());
    CPPUNIT_ASSERT(h.isValid());
    CPPUNIT_ASSERT(h.provenance()->moduleLabel() == label);
    CPPUNIT_ASSERT(h.provenance()->processName() == 
		   pProdConfig_->processName());
  }
}

void testeventprincipal::getbySelectorTest() 
{
  std::string label("fred");
  std::string instanceName("inst");
  put_a_product<edmtest::DummyProduct>(pProdConfig_, label, instanceName);

  edmtest::DummyProduct example;
  edm::TypeID tid(example);
  edm::ProcessNameSelector pnsel(pProdConfig_->processName());
  
  handle h = pEvent_->getBySelector(tid, pnsel);
  CPPUNIT_ASSERT(h.isValid());
  CPPUNIT_ASSERT(h.provenance()->processName() == 
		 pProdConfig_->processName());

  edm::ModuleLabelSelector mlsel(label);
  h = pEvent_->getBySelector(tid, mlsel);
  CPPUNIT_ASSERT(h.isValid());
  CPPUNIT_ASSERT(h.provenance()->moduleLabel() == label);

  edm::ProductInstanceNameSelector pinsel(instanceName);
  h = pEvent_->getBySelector(tid, pinsel);
  CPPUNIT_ASSERT(h.isValid());
  CPPUNIT_ASSERT(h.provenance()->productInstanceName() == instanceName);  

  // Warning: we don't actually expect the following selector to match
  // something, because I don't now have the time to form a proper
  // ModuleDescription and to get its ID. This only makes sure that
  // the class and its functions are instantiable.
  try
    {
      edm::ModuleDescription md;
      edm::ModuleDescriptionID mdid = md.id();
      edm::ModuleDescriptionSelector mdsel(mdid);
      h = pEvent_->getBySelector(tid, mdsel);
      assert("Failed to throw required exception!" == 0);
    }
  catch( edm::Exception& x )
    {
      // This is expected
    }
}

void testeventprincipal::processNameSelectorTest()
{
  // Put in the same product, from a few different process names.
  std::string wanted_process_name(pProdConfig_->processName());
  std::string label("fred");
  put_a_product<edmtest::DummyProduct>(pHltConfig_, label);
  put_a_product<edmtest::DummyProduct>(pProdConfig_, label);
  put_a_product<edmtest::DummyProduct>(pTestConfig_, label);
  put_a_product<edmtest::DummyProduct>(pUserConfig_, label);

  // Make sure we get back exactly one, from the right process.
  edmtest::DummyProduct example;
  edm::TypeID tid(example);
  edm::ProcessNameSelector pnsel(wanted_process_name);
  
  handle h = pEvent_->getBySelector(tid, pnsel);
  CPPUNIT_ASSERT(h.isValid());
  CPPUNIT_ASSERT(h.provenance()->moduleLabel() == label);
}

void testeventprincipal::getbyTypeTest() 
{
  std::string moduleLabel("fred");
  std::string productInstanceName("Rick");
  put_a_product<edmtest::DummyProduct>(pProdConfig_, moduleLabel, productInstanceName);
  
  edmtest::DummyProduct example;
  edm::TypeID tid(example);
  
  handle h = pEvent_->getByType(tid);
  CPPUNIT_ASSERT(h.isValid());
  CPPUNIT_ASSERT(h.provenance()->moduleLabel() == moduleLabel);
  CPPUNIT_ASSERT(h.provenance()->productInstanceName() == productInstanceName);
}

void testeventprincipal::getProvenanceTest() 
{
  std::string processName = "PROD";

  typedef edmtest::DummyProduct DP;
  typedef edm::Wrapper<DP> WDP;
  std::auto_ptr<DP> pr(new DP);
  std::auto_ptr<edm::EDProduct> pprod(new WDP(pr));
  std::auto_ptr<edm::Provenance> pprov(new edm::Provenance);
  std::string label("fred");

  edmtest::DummyProduct dp;
  edm::TypeID dummytype(dp);
  std::string className = dummytype.friendlyClassName();

  pprov->product.fullClassName_ = dummytype.userClassName();
  pprov->product.friendlyClassName_ = className;
  pprov->product.moduleLabel_ = label;
  pprov->product.processName_ = processName;
  pprov->product.init();

  pProductRegistry_->addProduct(pprov->product);
  pProductRegistry_->setProductIDs();

  edm::ProductRegistry::ProductList const& pl = pProductRegistry_->productList();
  edm::BranchKey const bk(pprov->product);
  edm::ProductRegistry::ProductList::const_iterator it = pl.find(bk);
  pprov->product.productID_ = it->second.productID_;

  edm::EventID col(1L);
  edm::Timestamp fakeTime;
  edm::EventPrincipal ep(col, fakeTime, *pProductRegistry_, *pProdConfig_);

  pEvent_->put(pprod, pprov);

  edm::ProductID id(1);

  edm::Provenance const& prov = pEvent_->getProvenance(id);
  CPPUNIT_ASSERT(prov.productID() == id);
}

void testeventprincipal::getAllProvenanceTest() 
{
  std::string processName = "PROD";

  typedef edmtest::DummyProduct DP;
  typedef edm::Wrapper<DP> WDP;
  std::auto_ptr<DP> pr(new DP);
  std::auto_ptr<edm::EDProduct> pprod(new WDP(pr));
  std::auto_ptr<edm::Provenance> pprov(new edm::Provenance);
  std::string label("fred");

  edmtest::DummyProduct dp;
  edm::TypeID dummytype(dp);
  std::string className = dummytype.friendlyClassName();

  pprov->product.fullClassName_ = dummytype.userClassName();
  pprov->product.friendlyClassName_ = className;
  pprov->product.moduleLabel_ = label;
  pprov->product.processName_ = processName;
  pprov->product.init();

  pProductRegistry_->addProduct(pprov->product);
  pProductRegistry_->setProductIDs();

  edm::ProductRegistry::ProductList const& pl = pProductRegistry_->productList();
  edm::BranchKey const bk(pprov->product);
  edm::ProductRegistry::ProductList::const_iterator it = pl.find(bk);
  pprov->product.productID_ = it->second.productID_;

  edm::EventID col(1L);
  edm::Timestamp fakeTime;
  edm::EventPrincipal ep(col, fakeTime, *pProductRegistry_, *pProdConfig_);

  pEvent_->put(pprod, pprov);

  edm::ProductID id(1);
  
  std::vector<edm::Provenance const*> provenances;

  pEvent_->getAllProvenance(provenances);
  CPPUNIT_ASSERT(provenances.size() == 1);
  CPPUNIT_ASSERT(provenances[0]->productID() == id);
}
#endif

