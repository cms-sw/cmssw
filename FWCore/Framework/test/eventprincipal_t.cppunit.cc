/*----------------------------------------------------------------------

Test of the EventPrincipal class.

$Id: eventprincipal_t.cppunit.cc,v 1.4 2005/07/21 17:25:38 wmtan Exp $

----------------------------------------------------------------------*/  
#include <cassert>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeinfo>

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/EDProduct/interface/EDP_ID.h"
#include "FWCore/Framework/interface/BasicHandle.h"
#include "FWCore/EDProduct/interface/Wrapper.h"
#include "FWCore/Framework/interface/Selector.h"
#include "FWCore/Framework/src/TypeID.h"
#include "FWCore/Framework/src/ToyProducts.h"

#include "FWCore/Framework/interface/EventPrincipal.h"
#include <cppunit/extensions/HelperMacros.h>

typedef edm::BasicHandle handle;

class testeventprincipal: public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(testeventprincipal);
CPPUNIT_TEST(failgetbyIdTest);
CPPUNIT_TEST(failgetbySelectorTest);
CPPUNIT_TEST(failgetbyLabelTest);
CPPUNIT_TEST(failgetManyTest);
CPPUNIT_TEST(failgetbyInvalidIdTest);
CPPUNIT_TEST(getbyIdTest);
CPPUNIT_TEST(getbySelectorTest);
CPPUNIT_TEST(getbyLabelTest);
CPPUNIT_TEST_SUITE_END();
public:
  void setUp(){}
  void tearDown(){}
  void failgetbyIdTest();
  void failgetbySelectorTest();
  void failgetbyLabelTest();
  void failgetManyTest();
  void failgetbyInvalidIdTest();
  void getbyIdTest();
  void getbySelectorTest();
  void getbyLabelTest();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testeventprincipal);

void testeventprincipal::failgetbyIdTest()
{
  edm::EventPrincipal ep;
  ep.addToProcessHistory("PROD");
  try
    {
      edm::EDP_ID id(1);
      handle h = ep.get(id);
      assert("Failed to throw required exception" == 0);
    }
  catch (edm::Exception& x)
    {
      // nothing to do
    }
  catch (...)
    {
      assert("Threw wrong kind of exception" == 0);
    }
}

void testeventprincipal::failgetbySelectorTest()
{
  edm::EventPrincipal ep;
  ep.addToProcessHistory("PROD");
  try
    {
      edm::TypeID tid(ep);   // sure not to match any product
      edm::ProcessNameSelector sel("PROD");
      handle h = ep.getBySelector(tid, sel);
      assert("Failed to throw required exception" == 0);      
    }
  catch (edm::Exception& x)
    {
      // nothing to do
    }
  catch (...)
    {
      assert("Threw wrong kind of exception" == 0);
    }
}

void testeventprincipal::failgetbyLabelTest()
{
  edm::EventPrincipal ep;
  ep.addToProcessHistory("PROD");
  try
    {
      edm::TypeID tid(ep);   // sure not to match any product
      std::string label("this does not exist");
      handle h = ep.getByLabel(tid, label, std::string());
      assert("Failed to throw required exception" == 0);      
    }
  catch (edm::Exception& x)
    {
      // nothing to do
    }
  catch (...)
    {
      assert("Threw wrong kind of exception" == 0);
    }
}

void testeventprincipal::failgetManyTest()
{
  edm::EventPrincipal ep;
  ep.addToProcessHistory("PROD");
  try
    {
      edm::TypeID tid(ep);   // sure not to match any product
      edm::ProcessNameSelector sel("PROD");
      std::vector<handle> handles;
      ep.getMany(tid, sel, handles);
      assert("Failed to throw required exception" == 0);      
    }
  catch (edm::Exception& x)
    {
      // nothing to do
    }
  catch (...)
    {
      assert("Threw wrong kind of exception" == 0);
    }

}

void testeventprincipal::failgetbyInvalidIdTest()
{
  edm::EventPrincipal ep;
  ep.addToProcessHistory("PROD");

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
  pprov->product.full_product_type_name = dummytype.userClassName();
  pprov->product.friendly_product_type_name = className;
  pprov->product.module.module_label = label;
  pprov->product.module.process_name = processName;
  pprov->product.init();
  ep.put(pprod, pprov);
  edm::EDP_ID id(1);

  try
    {
      handle h = ep.get(id-1);
      assert("Failed to throw required exception" == 0);      
    }
  catch (edm::Exception& x)
    {
      // nothing to do
    }
  catch (...)
    {
      assert("Threw wrong kind of exception" == 0);
    }
}


void testeventprincipal::getbyIdTest()
{
  std::string processName = "PROD";
  edm::EventPrincipal ep;
  ep.addToProcessHistory(processName);

  typedef edmtest::DummyProduct DP;
  typedef edm::Wrapper<DP> WDP;
  DP * pr = new DP;
  std::auto_ptr<edm::EDProduct> pprod(new WDP(*pr));
  std::auto_ptr<edm::Provenance> pprov(new edm::Provenance);
  std::string label("fred");

  edmtest::DummyProduct dp;
  edm::TypeID dummytype(dp);
  std::string className = dummytype.friendlyClassName();

  pprov->product.full_product_type_name = dummytype.userClassName();
  pprov->product.friendly_product_type_name = className;
  pprov->product.module.module_label = label;
  pprov->product.module.process_name = processName;
  pprov->product.init();
  ep.put(pprod, pprov);

  edm::EDP_ID id(1);
  
  try
    {
      handle h = ep.get(id);
      assert(h.isValid());
      assert(h.id() == id);
    }
  catch (edm::Exception& x)
    {
      std::cerr << x.what()<< std::endl;
      assert("Threw exception unexpectedly" == 0);
    }
  catch (...)
    {
      std::cerr << "Unknown exception type\n";
      assert("Threw exception unexpectedly" == 0);
    }
}

void testeventprincipal::getbyLabelTest()
{
  edm::EventPrincipal ep;
  std::string processName = "PROD";
  ep.addToProcessHistory(processName);

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

  pprov->product.full_product_type_name = dummytype.userClassName();
  pprov->product.friendly_product_type_name = className;


  pprov->product.module.module_label = label;
  pprov->product.product_instance_name = productInstanceName;
  pprov->product.module.process_name = processName;
  pprov->product.init();
  ep.put(pprod, pprov);
  
  try
    {
      edmtest::DummyProduct example;
      edm::TypeID tid(example);

      handle h = ep.getByLabel(tid, label, productInstanceName);
      assert(h.isValid());
      assert(h.provenance()->product.module.module_label == label);
    }
  catch (edm::Exception& x)
    {
      std::cerr << x.what()<< std::endl;
      assert("Threw exception unexpectedly" == 0);
    }
  catch (...)
    {
      std::cerr << "Unknown exception type\n";
      assert("Threw exception unexpectedly" == 0);
    }
}



void testeventprincipal::getbySelectorTest()
{
  edm::EventPrincipal ep;
  std::string processName("PROD");
  ep.addToProcessHistory(processName);

  typedef edmtest::DummyProduct DP;
  typedef edm::Wrapper<DP> WDP;
  DP * pr = new DP;
  std::auto_ptr<edm::EDProduct> pprod(new WDP(*pr));
  std::auto_ptr<edm::Provenance> pprov(new edm::Provenance);
  std::string label("fred");

  edmtest::DummyProduct dp;
  edm::TypeID dummytype(dp);
  std::string className = dummytype.friendlyClassName();

  pprov->product.full_product_type_name = dummytype.userClassName();
  pprov->product.friendly_product_type_name = className;

  pprov->product.module.module_label = label;
  pprov->product.module.process_name = processName;
  pprov->product.init();
  ep.put(pprod, pprov);

  try
    {
      edmtest::DummyProduct example;
      edm::TypeID tid(example);
      edm::ProcessNameSelector pnsel(processName);

      handle h = ep.getBySelector(tid, pnsel);
      assert(h.isValid());
      assert(h.provenance()->product.module.module_label == label);
    }
  catch (edm::Exception& x)
    {
      std::cerr << x.what()<< std::endl;
      assert("Threw exception unexpectedly" == 0);
    }
  catch (...)
    {
      std::cerr << "Unknown exception type\n";
      assert("Threw exception unexpectedly" == 0);
    }
}
