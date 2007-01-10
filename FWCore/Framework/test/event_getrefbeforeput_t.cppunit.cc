/*----------------------------------------------------------------------

Test of the EventPrincipal class.

$Id: event_getrefbeforeput_t.cppunit.cc,v 1.4 2006/12/05 23:56:18 paterno Exp $

----------------------------------------------------------------------*/  
#include <cassert>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeinfo>

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/GetPassID.h"
#include "FWCore/Utilities/interface/GetReleaseVersion.h"
#include "DataFormats/Common/interface/ProductID.h"
#include "FWCore/Framework/interface/BasicHandle.h"
#include "DataFormats/Common/interface/ProductRegistry.h"
#include "DataFormats/Common/interface/BranchDescription.h"
#include "DataFormats/Common/interface/Timestamp.h"
#include "DataFormats/Common/interface/Wrapper.h"
//#include "FWCore/Framework/interface/Selector.h"
#include "FWCore/Framework/interface/TypeID.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"

#include "FWCore/Framework/interface/EventPrincipal.h"

//have to do this evil in order to access commit_ member function
#define private public
#include "FWCore/Framework/interface/Event.h"
#undef private

#include <cppunit/extensions/HelperMacros.h>

class testEventGetRefBeforePut: public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE(testEventGetRefBeforePut);
CPPUNIT_TEST(failGetProductNotRegisteredTest);
CPPUNIT_TEST(getRefTest);
CPPUNIT_TEST_SUITE_END();
public:
  void setUp(){}
  void tearDown(){}
  void failGetProductNotRegisteredTest();
  void getRefTest();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testEventGetRefBeforePut);

void testEventGetRefBeforePut::failGetProductNotRegisteredTest() {

  edm::ProductRegistry preg;
  preg.setProductIDs();
  edm::EventID col(1L);
  edm::Timestamp fakeTime;
  edm::ProcessConfiguration pc("PROD", edm::ParameterSetID(), edm::getReleaseVersion(), edm::getPassID());
  edm::EventPrincipal ep(col, fakeTime, preg, pc);
  try {
     edm::ModuleDescription modDesc;
     modDesc.moduleName_ = "Blah";
     modDesc.moduleLabel_ = "blahs"; 
     edm::Event event(ep, modDesc);
     
     std::string label("this does not exist");
     edm::RefProd<edmtest::DummyProduct> ref = event.getRefBeforePut<edmtest::DummyProduct>(label);
     CPPUNIT_ASSERT("Failed to throw required exception" == 0);      
  }
  catch (edm::Exception& x) {
    // nothing to do
  }
  catch (...) {
    CPPUNIT_ASSERT("Threw wrong kind of exception" == 0);
  }
 
}

void testEventGetRefBeforePut::getRefTest() {
  std::string processName = "PROD";

  std::string label("fred");
  std::string productInstanceName("Rick");

  edmtest::IntProduct dp;
  edm::TypeID dummytype(dp);
  std::string className = dummytype.friendlyClassName();

  std::auto_ptr<edm::Provenance> pprov(new edm::Provenance);

  pprov->product.fullClassName_ = dummytype.userClassName();
  pprov->product.friendlyClassName_ = className;

  edm::ModuleDescription modDesc;
  modDesc.moduleName_ = "Blah";

  pprov->product.moduleLabel_ = label;
  pprov->product.productInstanceName_ = productInstanceName;
  pprov->product.processName_ = processName;
  pprov->product.moduleDescriptionID_ = modDesc.id();
  pprov->product.init();

  edm::ProductRegistry preg;
  preg.addProduct(pprov->product);
  preg.setProductIDs();
  edm::EventID col(1L);
  edm::Timestamp fakeTime;
  edm::ProcessConfiguration pc(processName, edm::ParameterSetID(), edm::getReleaseVersion(), edm::getPassID());
  edm::EventPrincipal ep(col, fakeTime, preg, pc);

  edm::RefProd<edmtest::IntProduct> refToProd;
  try {
    edm::ModuleDescription modDesc;
    modDesc.moduleName_="Blah";
    modDesc.moduleLabel_=label; 
    modDesc.processConfiguration_ = pc;

    edm::Event event(ep, modDesc);
    std::auto_ptr<edmtest::IntProduct> pr(new edmtest::IntProduct);
    pr->value = 10;
    
    refToProd = event.getRefBeforePut<edmtest::IntProduct>(productInstanceName);
    event.put(pr,productInstanceName);
    event.commit_();
  }
  catch (cms::Exception& x) {
    std::cerr << x.explainSelf()<< std::endl;
    CPPUNIT_ASSERT("Threw exception unexpectedly" == 0);
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
  CPPUNIT_ASSERT(refToProd->value == 10);
}

