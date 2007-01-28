/*----------------------------------------------------------------------

Test of the EventPrincipal class.

$Id: generichandle_t.cppunit.cc,v 1.15 2007/01/10 05:59:30 wmtan Exp $

----------------------------------------------------------------------*/  
#include <string>

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/GetPassID.h"
#include "FWCore/Utilities/interface/GetReleaseVersion.h"
#include "DataFormats/Common/interface/ProductRegistry.h"
#include "DataFormats/Common/interface/ModuleDescription.h"
#include "DataFormats/Common/interface/Timestamp.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "FWCore/Framework/interface/TypeID.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"

#include "FWCore/Framework/interface/GenericHandle.h"
#include <cppunit/extensions/HelperMacros.h>

class testGenericHandle: public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE(testGenericHandle);
CPPUNIT_TEST(failgetbyLabelTest);
CPPUNIT_TEST(getbyLabelTest);
CPPUNIT_TEST(failWrongType);
CPPUNIT_TEST_SUITE_END();
public:
  void setUp(){}
  void tearDown(){}
  void failgetbyLabelTest();
  void failWrongType();
  void getbyLabelTest();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testGenericHandle);

void testGenericHandle::failWrongType() {
   try {
      //intentionally misspelled type
      edm::GenericHandle h("edmtest::DmmyProduct");
      CPPUNIT_ASSERT("Failed to thow"==0);
   }
   catch (edm::Exception& x) {
      // nothing to do
   }
   catch (...) {
      CPPUNIT_ASSERT("Threw wrong kind of exception" == 0);
   }
}
void testGenericHandle::failgetbyLabelTest() {

  edm::EventID id;
  edm::Timestamp time;
  edm::ProductRegistry preg;
  edm::ProcessConfiguration pc("PROD", edm::ParameterSetID(), edm::getReleaseVersion(), edm::getPassID());
  edm::EventPrincipal ep(id, time, preg, pc);
  edm::GenericHandle h("edmtest::DummyProduct");
  try {
     edm::ModuleDescription modDesc;
     modDesc.moduleName_="Blah";
     modDesc.moduleLabel_="blahs"; 
     edm::Event event(ep, modDesc);
     
     std::string label("this does not exist");
     event.getByLabel(label,h);
    CPPUNIT_ASSERT("Failed to throw required exception" == 0);      
  }
  catch (edm::Exception& x) {
    // nothing to do
  }
  catch (...) {
    CPPUNIT_ASSERT("Threw wrong kind of exception" == 0);
  }
 
}

void testGenericHandle::getbyLabelTest() {
  std::string processName = "PROD";

  typedef edmtest::DummyProduct DP;
  typedef edm::Wrapper<DP> WDP;
  std::auto_ptr<DP> pr(new DP);
  std::auto_ptr<edm::EDProduct> pprod(new WDP(pr));
  std::string label("fred");
  std::string productInstanceName("Rick");

  edmtest::DummyProduct dp;
  edm::TypeID dummytype(dp);
  std::string className = dummytype.friendlyClassName();

  edm::BranchDescription product;

  product.fullClassName_ = dummytype.userClassName();
  product.friendlyClassName_ = className;

  edm::ModuleDescription modDesc;
  modDesc.moduleName_ = "Blah";

  product.moduleLabel_ = label;
  product.productInstanceName_ = productInstanceName;
  product.processName_ = processName;
  product.moduleDescriptionID_ = modDesc.id();
  product.init();

  edm::ProductRegistry preg;
  preg.addProduct(product);
  preg.setProductIDs();

  edm::ProductRegistry::ProductList const& pl = preg.productList();
  edm::BranchKey const bk(product);
  edm::ProductRegistry::ProductList::const_iterator it = pl.find(bk);
  product.productID_ = it->second.productID_;

  edm::EventID col(1L);
  edm::Timestamp fakeTime;
  edm::ProcessConfiguration pc("PROD", edm::ParameterSetID(), edm::getReleaseVersion(), edm::getPassID());
  edm::EventPrincipal ep(col, fakeTime, preg, pc);

  std::auto_ptr<edm::Provenance> pprov(new edm::Provenance(product, edm::BranchEntryDescription::Success));
  ep.put(pprod, pprov);
  
  edm::GenericHandle h("edmtest::DummyProduct");
  try {
    edm::ModuleDescription modDesc;
    modDesc.moduleName_="Blah";
    modDesc.moduleLabel_="blahs"; 
    edm::Event event(ep, modDesc);

    event.getByLabel(label, productInstanceName,h);
  }
  catch (cms::Exception& x) {
    std::cerr << x.explainSelf()<< std::endl;
    CPPUNIT_ASSERT("Threw cms::Exception unexpectedly" == 0);
  }
  catch(seal::Error& x){
     std::cerr <<x.explainSelf()<<std::endl;
     CPPUNIT_ASSERT("Threw seal Error"==0);
  }
  catch(std::exception& x){
     std::cerr <<x.what()<<std::endl;
     CPPUNIT_ASSERT("threw std::exception"==0);
  }
  catch (...) {
    std::cerr << "Unknown exception type\n";
    CPPUNIT_ASSERT("Threw exception unexpectedly" == 0);
  }
  CPPUNIT_ASSERT(h.isValid());
  CPPUNIT_ASSERT(h.provenance()->moduleLabel() == label);
}

