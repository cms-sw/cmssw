/*----------------------------------------------------------------------

Test of the EventPrincipal class.

$Id: eventprincipal_t.cppunit.cc,v 1.30 2006/11/15 00:19:13 paterno Exp $

----------------------------------------------------------------------*/  
#include <memory>
#include <stdexcept>
#include <string>
#include <typeinfo>


#include "DataFormats/Common/interface/BranchDescription.h"
#include "DataFormats/Common/interface/ParameterSetID.h"
#include "DataFormats/Common/interface/ProductID.h"
#include "DataFormats/Common/interface/ProductRegistry.h"
#include "DataFormats/Common/interface/Timestamp.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Framework/interface/BasicHandle.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/Selector.h"
#include "FWCore/Framework/interface/TypeID.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/GetPassID.h"
#include "FWCore/Utilities/interface/GetReleaseVersion.h"
#include "FWCore/Utilities/interface/PretendToUse.h"
#include <cppunit/extensions/HelperMacros.h>

typedef edm::BasicHandle handle;

class testeventprincipal: public CppUnit::TestFixture 
{
  CPPUNIT_TEST_SUITE(testeventprincipal);
  CPPUNIT_TEST_EXCEPTION(failgetbyIdTest, edm::Exception);
  CPPUNIT_TEST_EXCEPTION(failgetbySelectorTest, edm::Exception);
  CPPUNIT_TEST_EXCEPTION(failgetbyLabelTest, edm::Exception);
  CPPUNIT_TEST_EXCEPTION(failgetManyTest, edm::Exception);
  CPPUNIT_TEST_EXCEPTION(failgetbyTypeTest, edm::Exception);
  CPPUNIT_TEST_EXCEPTION(failgetManybyTypeTest, edm::Exception);
  CPPUNIT_TEST_EXCEPTION(failgetbyInvalidIdTest, edm::Exception);
  CPPUNIT_TEST_EXCEPTION(failgetProvenanceTest, edm::Exception);
  CPPUNIT_TEST(getbyIdTest);
  CPPUNIT_TEST(getbySelectorTest);
  CPPUNIT_TEST(getbyLabelTest);
  CPPUNIT_TEST(getbyTypeTest);
  CPPUNIT_TEST(getProvenanceTest);
  CPPUNIT_TEST(getAllProvenanceTest);
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
  void getbyIdTest();
  void getbySelectorTest();
  void processNameSelectorTest();
  void getbyLabelTest();
  void getbyTypeTest();
  void getProvenanceTest();
  void getAllProvenanceTest();

 private:

  template <class PRODUCT_TYPE>
  void put_a_product(std::string const& processName,
		     std::string const& moduleLabel,
		     std::string const& productInstanceName = std::string() );


  edm::ProcessConfiguration* pConfig_;  
  edm::ProductRegistry*      pProductRegistry_;
  edm::EventPrincipal*       pEvent_;

  edm::EventID               eventID_;
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testeventprincipal);

void testeventprincipal::setUp()
{
  // Making a functional EventPrincipal is not trivial, so we do it here...
  eventID_ = edm::EventID(101, 20, false);
  pConfig_ = new edm::ProcessConfiguration("PROD", 
					   edm::ParameterSetID(), 
					   edm::getReleaseVersion(), 
					   edm::getPassID());
  pProductRegistry_ = new edm::ProductRegistry;
  edm::Timestamp now(1234567UL);
  pEvent_  = new edm::EventPrincipal(eventID_, now, *pProductRegistry_);
}

void testeventprincipal::tearDown()
{
  // in case of error in CPPUNIT code, clear pointers...
  delete pEvent_;
  pEvent_  = 0;

  delete pProductRegistry_;
  pProductRegistry_ = 0;

  delete pConfig_;
  pConfig_          = 0;
}

template <class PRODUCT_TYPE>
void testeventprincipal::put_a_product(std::string const& processName,
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
  provenance->product.processName_         = processName;
  provenance->product.productInstanceName_ = productInstanceName;
  provenance->product.init();

  pProductRegistry_->addProduct(provenance->product);
  pProductRegistry_->setProductIDs();

  edm::ProductRegistry::ProductList const& pl = pProductRegistry_->productList();
  edm::BranchKey const bk(provenance->product);
  edm::ProductRegistry::ProductList::const_iterator it = pl.find(bk);
  provenance->product.productID_ = it->second.productID_;

  edm::EventID col(1L);
  edm::Timestamp fakeTime;
  edm::EventPrincipal ep(col, fakeTime, *pProductRegistry_);
  pEvent_->addToProcessHistory(*pConfig_);
  pEvent_->put(product, provenance);
}

//----------------------------------------------------------------------
// Test functions
//----------------------------------------------------------------------

void testeventprincipal::failgetbyIdTest() 
{
  pEvent_->addToProcessHistory(*pConfig_);
  edm::ProductID id;
  handle h = pEvent_->get(id);
  pretendToUse(h);
}

void testeventprincipal::failgetbySelectorTest()
{
  pEvent_->addToProcessHistory(*pConfig_);

  edm::TypeID tid(*pEvent_);   // sure not to match any product
  edm::ProcessNameSelector sel("PROD");
  handle h = pEvent_->getBySelector(tid, sel);
  pretendToUse(h);  
}

void testeventprincipal::failgetbyLabelTest() 
{
  pEvent_->addToProcessHistory(*pConfig_);
  edm::TypeID tid(*pEvent_);   // sure not to match any product
  std::string label("this does not exist");
  handle h = pEvent_->getByLabel(tid, label, std::string());
  pretendToUse(h);
}

void testeventprincipal::failgetManyTest() 
{
  pEvent_->addToProcessHistory(*pConfig_);
  edm::TypeID tid(*pEvent_);   // sure not to match any product
  edm::ProcessNameSelector sel("PROD");
  std::vector<handle> handles;
  pEvent_->getMany(tid, sel, handles);
}

void testeventprincipal::failgetbyTypeTest() 
{
  pEvent_->addToProcessHistory(*pConfig_);

  edm::TypeID tid(*pEvent_);   // sure not to match any product
  handle h = pEvent_->getByType(tid);
  pretendToUse(h);
}

void testeventprincipal::failgetManybyTypeTest() 
{
  pEvent_->addToProcessHistory(*pConfig_);

  edm::TypeID tid(*pEvent_);   // sure not to match any product
  std::vector<handle> handles;
  pEvent_->getManyByType(tid, handles);
}

void testeventprincipal::failgetbyInvalidIdTest() 
{
   std::string processName = "PROD";
   std::string label("fred");
   put_a_product<edmtest::DummyProduct>(processName, label);

  edm::ProductID id;

  handle h = pEvent_->get(id);
  pretendToUse(h);
}

void testeventprincipal::failgetProvenanceTest() 
{
  pEvent_->addToProcessHistory(*pConfig_);

  edm::ProductID id;
  edm::Provenance const& prov = pEvent_->getProvenance(id);
  pretendToUse(prov);
}


void testeventprincipal::getbyIdTest() 
{
  std::string processName = "PROD";
  std::string label("modulename");
  put_a_product<edmtest::DummyProduct>(processName, label);
  edm::ProductID id(1);
  handle h = pEvent_->get(id);
  CPPUNIT_ASSERT(h.isValid());
  CPPUNIT_ASSERT(h.id() == id);
}

void testeventprincipal::getbyLabelTest() 
{
  std::string processName = "PROD";
  std::string label("fred");
  std::string productInstanceName("Rick");
  put_a_product<edmtest::DummyProduct>(processName, label, productInstanceName);
  
  edmtest::DummyProduct example;
  edm::TypeID tid(example);

  handle h = pEvent_->getByLabel(tid, label, productInstanceName);
  CPPUNIT_ASSERT(h.isValid());
  CPPUNIT_ASSERT(h.provenance()->moduleLabel() == label);
  {
    handle h = pEvent_->getByLabel(tid, label, productInstanceName, processName);
    CPPUNIT_ASSERT(h.isValid());
    CPPUNIT_ASSERT(h.provenance()->moduleLabel() == label);
    CPPUNIT_ASSERT(h.provenance()->processName() == processName);
  }
}

void testeventprincipal::getbySelectorTest() 
{
  std::string processName("PROD");
  std::string label("fred");
  std::string instanceName("inst");
  put_a_product<edmtest::DummyProduct>(processName, label, instanceName);

  edmtest::DummyProduct example;
  edm::TypeID tid(example);
  edm::ProcessNameSelector pnsel(processName);
  
  handle h = pEvent_->getBySelector(tid, pnsel);
  CPPUNIT_ASSERT(h.isValid());
  CPPUNIT_ASSERT(h.provenance()->processName() == processName);

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
  std::string wanted_process_name("PROD");
  std::string label("fred");
  put_a_product<edmtest::DummyProduct>("HLT", label);
  put_a_product<edmtest::DummyProduct>(wanted_process_name, label);
  put_a_product<edmtest::DummyProduct>("TEST", label);
  put_a_product<edmtest::DummyProduct>("USER", label);

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
  std::string processName = "PROD";
  std::string moduleLabel("fred");
  std::string productInstanceName("Rick");
  put_a_product<edmtest::DummyProduct>(processName, moduleLabel, productInstanceName);
  
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
  edm::EventPrincipal ep(col, fakeTime, *pProductRegistry_);
  pEvent_->addToProcessHistory(*pConfig_);

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
  edm::EventPrincipal ep(col, fakeTime, *pProductRegistry_);
  pEvent_->addToProcessHistory(*pConfig_);

  pEvent_->put(pprod, pprov);

  edm::ProductID id(1);
  
  std::vector<edm::Provenance const*> provenances;

  pEvent_->getAllProvenance(provenances);
  CPPUNIT_ASSERT(provenances.size() == 1);
  CPPUNIT_ASSERT(provenances[0]->productID() == id);
}
