/*----------------------------------------------------------------------

Test of the EventPrincipal class.

$Id: eventprincipal_t.cppunit.cc,v 1.15 2005/09/07 19:15:56 wmtan Exp $

----------------------------------------------------------------------*/  
#include <cassert>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeinfo>

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/EDProduct/interface/ProductID.h"
#include "FWCore/Framework/interface/BasicHandle.h"
#include "FWCore/Framework/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/ProductDescription.h"
#include "FWCore/EDProduct/interface/Timestamp.h"
#include "FWCore/EDProduct/interface/Wrapper.h"
#include "FWCore/Framework/interface/Selector.h"
#include "FWCore/Framework/src/TypeID.h"
#include "FWCore/Framework/src/ToyProducts.h"

#include "FWCore/Framework/interface/EventPrincipal.h"
#include <cppunit/extensions/HelperMacros.h>

typedef edm::BasicHandle handle;

class testeventprincipal: public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE(testeventprincipal);
CPPUNIT_TEST(failgetbyIdTest);
CPPUNIT_TEST(failgetbySelectorTest);
CPPUNIT_TEST(failgetbyLabelTest);
CPPUNIT_TEST(failgetManyTest);
CPPUNIT_TEST(failgetbyTypeTest);
CPPUNIT_TEST(failgetManybyTypeTest);
CPPUNIT_TEST(failgetbyInvalidIdTest);
CPPUNIT_TEST(failgetProvenanceTest);
CPPUNIT_TEST(getbyIdTest);
CPPUNIT_TEST(getbySelectorTest);
CPPUNIT_TEST(getbyLabelTest);
CPPUNIT_TEST(getbyTypeTest);
CPPUNIT_TEST(getProvenanceTest);
CPPUNIT_TEST(getAllProvenanceTest);
CPPUNIT_TEST_SUITE_END();
public:
  void setUp(){}
  void tearDown(){}
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
  void getbyLabelTest();
  void getbyTypeTest();
  void getProvenanceTest();
  void getAllProvenanceTest();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testeventprincipal);

void testeventprincipal::failgetbyIdTest() {
  edm::EventPrincipal ep;
  ep.addToProcessHistory("PROD");
  try {
    edm::ProductID id;
    handle h = ep.get(id);
    assert("Failed to throw required exception" == 0);
  }
  catch (edm::Exception& x) {
    // nothing to do
  }
  catch (...) {
    assert("Threw wrong kind of exception" == 0);
  }
}

void testeventprincipal::failgetbySelectorTest()
{
  edm::EventPrincipal ep;
  ep.addToProcessHistory("PROD");
  try {
    edm::TypeID tid(ep);   // sure not to match any product
    edm::ProcessNameSelector sel("PROD");
    handle h = ep.getBySelector(tid, sel);
    assert("Failed to throw required exception" == 0);      
  }
  catch (edm::Exception& x) {
    // nothing to do
  }
  catch (...) {
    assert("Threw wrong kind of exception" == 0);
  }
}

void testeventprincipal::failgetbyLabelTest() {
  edm::EventPrincipal ep;
  ep.addToProcessHistory("PROD");
  try {
    edm::TypeID tid(ep);   // sure not to match any product
    std::string label("this does not exist");
    handle h = ep.getByLabel(tid, label, std::string());
    assert("Failed to throw required exception" == 0);      
  }
  catch (edm::Exception& x) {
    // nothing to do
  }
  catch (...) {
    assert("Threw wrong kind of exception" == 0);
  }
}

void testeventprincipal::failgetManyTest() {
  edm::EventPrincipal ep;
  ep.addToProcessHistory("PROD");
  try {
    edm::TypeID tid(ep);   // sure not to match any product
    edm::ProcessNameSelector sel("PROD");
    std::vector<handle> handles;
    ep.getMany(tid, sel, handles);
    assert("Failed to throw required exception" == 0);      
  }
  catch (edm::Exception& x) {
    // nothing to do
  }
  catch (...) {
    assert("Threw wrong kind of exception" == 0);
  }
}

void testeventprincipal::failgetbyTypeTest() {
  edm::EventPrincipal ep;
  ep.addToProcessHistory("PROD");
  try {
    edm::TypeID tid(ep);   // sure not to match any product
    handle h = ep.getByType(tid);
    assert("Failed to throw required exception" == 0);      
  }
  catch (edm::Exception& x) {
    // nothing to do
  }
  catch (...) {
    assert("Threw wrong kind of exception" == 0);
  }
}

void testeventprincipal::failgetManybyTypeTest() {
  edm::EventPrincipal ep;
  ep.addToProcessHistory("PROD");
  try {
    edm::TypeID tid(ep);   // sure not to match any product
    std::vector<handle> handles;
    ep.getManyByType(tid, handles);
    assert("Failed to throw required exception" == 0);      
  }
  catch (edm::Exception& x) {
    // nothing to do
  }
  catch (...) {
    assert("Threw wrong kind of exception" == 0);
  }

}

void testeventprincipal::failgetbyInvalidIdTest() {
  typedef edmtest::DummyProduct DP;
  typedef edm::Wrapper<DP> WDP;
  DP * pr = new DP;
  std::auto_ptr<edm::EDProduct> pprod(new WDP(*pr));
  std::auto_ptr<edm::Provenance> pprov(new edm::Provenance);
  std::string label("fred");
  std::string processName = "PROD";
  
  edmtest::DummyProduct dp;
  edm::TypeID dummytype(dp);
  std::string className = dummytype.friendlyClassName();
  pprov->product.fullClassName_ = dummytype.userClassName();
  pprov->product.friendlyClassName_ = className;
  pprov->product.module.moduleLabel_ = label;
  pprov->product.module.processName_ = processName;
  pprov->product.init();

  edm::ProductRegistry preg;
  preg.addProduct(pprov->product);
  preg.setProductIDs();
  edm::EventID col(1L);
  edm::Timestamp fakeTime;
  edm::EventPrincipal ep(col, fakeTime, preg);
  ep.addToProcessHistory("PROD");

  ep.put(pprod, pprov);

  edm::ProductID id;

  try {
    handle h = ep.get(id);
    assert("Failed to throw required exception" == 0);      
  }
  catch (edm::Exception& x) {
    // nothing to do
  }
  catch (...) {
    assert("Threw wrong kind of exception" == 0);
  }
}

void testeventprincipal::failgetProvenanceTest() {
  edm::EventPrincipal ep;
  ep.addToProcessHistory("PROD");
  try {
    edm::ProductID id;
    edm::Provenance const& prov = ep.getProvenance(id);
    assert("Failed to throw required exception" == 0);
    assert(prov.product.productID_ == id); // Avoids unused variable build warning 
  }
  catch (edm::Exception& x) {
    // nothing to do
  }
  catch (...) {
    assert("Threw wrong kind of exception" == 0);
  }
}


void testeventprincipal::getbyIdTest() {
  std::string processName = "PROD";

  typedef edmtest::DummyProduct DP;
  typedef edm::Wrapper<DP> WDP;
  DP * pr = new DP;
  std::auto_ptr<edm::EDProduct> pprod(new WDP(*pr));
  std::auto_ptr<edm::Provenance> pprov(new edm::Provenance);
  std::string label("fred");

  edmtest::DummyProduct dp;
  edm::TypeID dummytype(dp);
  std::string className = dummytype.friendlyClassName();

  pprov->product.fullClassName_ = dummytype.userClassName();
  pprov->product.friendlyClassName_ = className;
  pprov->product.module.moduleLabel_ = label;
  pprov->product.module.processName_ = processName;
  pprov->product.init();

  edm::ProductRegistry preg;
  preg.addProduct(pprov->product);
  preg.setProductIDs();
  edm::EventID col(1L);
  edm::Timestamp fakeTime;
  edm::EventPrincipal ep(col, fakeTime, preg);
  ep.addToProcessHistory("PROD");

  ep.put(pprod, pprov);

  edm::ProductID id(1);
  
  try {
    handle h = ep.get(id);
    assert(h.isValid());
    assert(h.id() == id);
  }
  catch (edm::Exception& x) {
    std::cerr << x.what()<< std::endl;
    assert("Threw exception unexpectedly" == 0);
  }
  catch (...) {
    std::cerr << "Unknown exception type\n";
    assert("Threw exception unexpectedly" == 0);
  }
}

void testeventprincipal::getbyLabelTest() {
  std::string processName = "PROD";

  typedef edmtest::DummyProduct DP;
  typedef edm::Wrapper<DP> WDP;
  DP * pr = new DP;
  std::auto_ptr<edm::EDProduct> pprod(new WDP(*pr));
  std::auto_ptr<edm::Provenance> pprov(new edm::Provenance);
  std::string label("fred");
  std::string productInstanceName("Rick");

  edmtest::DummyProduct dp;
  edm::TypeID dummytype(dp);
  std::string className = dummytype.friendlyClassName();

  pprov->product.fullClassName_ = dummytype.userClassName();
  pprov->product.friendlyClassName_ = className;


  pprov->product.module.moduleLabel_ = label;
  pprov->product.productInstanceName_ = productInstanceName;
  pprov->product.module.processName_ = processName;
  pprov->product.init();

  edm::ProductRegistry preg;
  preg.addProduct(pprov->product);
  preg.setProductIDs();
  edm::EventID col(1L);
  edm::Timestamp fakeTime;
  edm::EventPrincipal ep(col, fakeTime, preg);
  ep.addToProcessHistory("PROD");

  ep.put(pprod, pprov);
  
  try {
    edmtest::DummyProduct example;
    edm::TypeID tid(example);

    handle h = ep.getByLabel(tid, label, productInstanceName);
    assert(h.isValid());
    assert(h.provenance()->product.module.moduleLabel_ == label);
  }
  catch (edm::Exception& x) {
    std::cerr << x.what()<< std::endl;
    assert("Threw exception unexpectedly" == 0);
  }
  catch (...) {
    std::cerr << "Unknown exception type\n";
    assert("Threw exception unexpectedly" == 0);
  }
}

void testeventprincipal::getbySelectorTest() {
  std::string processName("PROD");

  typedef edmtest::DummyProduct DP;
  typedef edm::Wrapper<DP> WDP;
  DP * pr = new DP;
  std::auto_ptr<edm::EDProduct> pprod(new WDP(*pr));
  std::auto_ptr<edm::Provenance> pprov(new edm::Provenance);
  std::string label("fred");

  edmtest::DummyProduct dp;
  edm::TypeID dummytype(dp);
  std::string className = dummytype.friendlyClassName();

  pprov->product.fullClassName_ = dummytype.userClassName();
  pprov->product.friendlyClassName_ = className;

  pprov->product.module.moduleLabel_ = label;
  pprov->product.module.processName_ = processName;
  pprov->product.init();

  edm::ProductRegistry preg;
  preg.addProduct(pprov->product);
  preg.setProductIDs();
  edm::EventID col(1L);
  edm::Timestamp fakeTime;
  edm::EventPrincipal ep(col, fakeTime, preg);
  ep.addToProcessHistory("PROD");

  ep.put(pprod, pprov);

  try {
    edmtest::DummyProduct example;
    edm::TypeID tid(example);
    edm::ProcessNameSelector pnsel(processName);

    handle h = ep.getBySelector(tid, pnsel);
    assert(h.isValid());
    assert(h.provenance()->product.module.moduleLabel_ == label);
  }
  catch (edm::Exception& x) {
    std::cerr << x.what()<< std::endl;
    assert("Threw exception unexpectedly" == 0);
  }
  catch (...) {
    std::cerr << "Unknown exception type\n";
    assert("Threw exception unexpectedly" == 0);
  }
}

void testeventprincipal::getbyTypeTest() {
  std::string processName = "PROD";

  typedef edmtest::DummyProduct DP;
  typedef edm::Wrapper<DP> WDP;
  DP * pr = new DP;
  std::auto_ptr<edm::EDProduct> pprod(new WDP(*pr));
  std::auto_ptr<edm::Provenance> pprov(new edm::Provenance);
  std::string label("fred");
  std::string productInstanceName("Rick");

  edmtest::DummyProduct dp;
  edm::TypeID dummytype(dp);
  std::string className = dummytype.friendlyClassName();

  pprov->product.fullClassName_ = dummytype.userClassName();
  pprov->product.friendlyClassName_ = className;


  pprov->product.module.moduleLabel_ = label;
  pprov->product.productInstanceName_ = productInstanceName;
  pprov->product.module.processName_ = processName;
  pprov->product.init();

  edm::ProductRegistry preg;
  preg.addProduct(pprov->product);
  preg.setProductIDs();
  edm::EventID col(1L);
  edm::Timestamp fakeTime;
  edm::EventPrincipal ep(col, fakeTime, preg);
  ep.addToProcessHistory("PROD");

  ep.put(pprod, pprov);
  
  try {
    edmtest::DummyProduct example;
    edm::TypeID tid(example);

    handle h = ep.getByType(tid);
    assert(h.isValid());
    assert(h.provenance()->product.module.moduleLabel_ == label);
  }
  catch (edm::Exception& x) {
    std::cerr << x.what()<< std::endl;
    assert("Threw exception unexpectedly" == 0);
  }
  catch (...) {
    std::cerr << "Unknown exception type\n";
    assert("Threw exception unexpectedly" == 0);
  }
}

void testeventprincipal::getProvenanceTest() {
  std::string processName = "PROD";

  typedef edmtest::DummyProduct DP;
  typedef edm::Wrapper<DP> WDP;
  DP * pr = new DP;
  std::auto_ptr<edm::EDProduct> pprod(new WDP(*pr));
  std::auto_ptr<edm::Provenance> pprov(new edm::Provenance);
  std::string label("fred");

  edmtest::DummyProduct dp;
  edm::TypeID dummytype(dp);
  std::string className = dummytype.friendlyClassName();

  pprov->product.fullClassName_ = dummytype.userClassName();
  pprov->product.friendlyClassName_ = className;
  pprov->product.module.moduleLabel_ = label;
  pprov->product.module.processName_ = processName;
  pprov->product.init();

  edm::ProductRegistry preg;
  preg.addProduct(pprov->product);
  preg.setProductIDs();
  edm::EventID col(1L);
  edm::Timestamp fakeTime;
  edm::EventPrincipal ep(col, fakeTime, preg);
  ep.addToProcessHistory("PROD");

  ep.put(pprod, pprov);

  edm::ProductID id(1);
  
  try {
    edm::Provenance const& prov = ep.getProvenance(id);
    assert(prov.product.productID_ == id);
  }
  catch (edm::Exception& x) {
    std::cerr << x.what()<< std::endl;
    assert("Threw exception unexpectedly" == 0);
  }
  catch (...) {
    std::cerr << "Unknown exception type\n";
    assert("Threw exception unexpectedly" == 0);
  }
}

void testeventprincipal::getAllProvenanceTest() {
  std::string processName = "PROD";

  typedef edmtest::DummyProduct DP;
  typedef edm::Wrapper<DP> WDP;
  DP * pr = new DP;
  std::auto_ptr<edm::EDProduct> pprod(new WDP(*pr));
  std::auto_ptr<edm::Provenance> pprov(new edm::Provenance);
  std::string label("fred");

  edmtest::DummyProduct dp;
  edm::TypeID dummytype(dp);
  std::string className = dummytype.friendlyClassName();

  pprov->product.fullClassName_ = dummytype.userClassName();
  pprov->product.friendlyClassName_ = className;
  pprov->product.module.moduleLabel_ = label;
  pprov->product.module.processName_ = processName;
  pprov->product.init();

  edm::ProductRegistry preg;
  preg.addProduct(pprov->product);
  preg.setProductIDs();
  edm::EventID col(1L);
  edm::Timestamp fakeTime;
  edm::EventPrincipal ep(col, fakeTime, preg);
  ep.addToProcessHistory("PROD");

  ep.put(pprod, pprov);

  edm::ProductID id(1);
  
  std::vector<edm::Provenance const*> provenances;
  try {
    ep.getAllProvenance(provenances);
    std::cout << provenances.size() << std::endl;
    assert(provenances.size() == 1);
    assert(provenances[0]->product.productID_ == id);
  }
  catch (edm::Exception& x) {
    std::cerr << x.what()<< std::endl;
    assert("Threw exception unexpectedly" == 0);
  }
  catch (...) {
    std::cerr << "Unknown exception type\n";
    assert("Threw exception unexpectedly" == 0);
  }
}
