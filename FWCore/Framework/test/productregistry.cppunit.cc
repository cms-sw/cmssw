/**
   \file
   test for ProductRegistry 

   \author Stefano ARGIRO
   \version $Id: productregistry.cppunit.cc,v 1.4 2005/10/03 19:02:41 wmtan Exp $
   \date 21 July 2005
*/


#include <iostream>
#include <cppunit/extensions/HelperMacros.h>
#include "FWCore/Framework/interface/EventProcessor.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/Framework/src/SignallingProductRegistry.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/Framework/interface/ModuleDescription.h"
#include "FWCore/Utilities/interface/ProblemTracker.h"


class testProductRegistry: public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(testProductRegistry);

CPPUNIT_TEST(testSignal);
CPPUNIT_TEST(testWatch);
CPPUNIT_TEST_EXCEPTION(testCircular,cms::Exception);

CPPUNIT_TEST(testProductRegistration);

CPPUNIT_TEST_SUITE_END();

public:
  void setUp(){}
  void tearDown(){}
  void testSignal();
  void testWatch();
  void testCircular();
  void testProductRegistration();

};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testProductRegistry);

namespace {
   struct Listener {
      int* heard_;
      Listener(int& hear) :heard_(&hear) {}
      void operator()(const edm::BranchDescription&){
         ++(*heard_);
      }
   };

   struct Responder {
      std::string name_;
      edm::ProductRegistry* reg_;
      Responder(const std::string& iName,
                edm::ConstProductRegistry& iConstReg,
                edm::ProductRegistry& iReg):name_(iName),reg_(&iReg)
      {
        iConstReg.watchProductAdditions(this, &Responder::respond);
      }
      void respond(const edm::BranchDescription& iDesc){
         edm::ModuleDescription modDesc;
         modDesc.moduleLabel_ = name_;
         edm::BranchDescription prod(iDesc);
         prod.productInstanceName_ = prod.productInstanceName_+"-"+prod.module.moduleLabel_;
         prod.module = modDesc;
         reg_->addProduct(prod);
      }
   };
}

void  testProductRegistry:: testSignal(){
   using namespace edm;
   SignallingProductRegistry reg;
   
   int hear=0;
   Listener listening(hear);
   reg.productAddedSignal_.connect(listening);
   
   ModuleDescription modDesc;
   BranchDescription prod(modDesc, "int", "int", "int",0);
   
   reg.addProduct(prod);
   CPPUNIT_ASSERT(1==hear);
}

void  testProductRegistry:: testWatch(){
   using namespace edm;
   SignallingProductRegistry reg;
   ConstProductRegistry constReg(reg);
   
   int hear=0;
   Listener listening(hear);
   constReg.watchProductAdditions(listening);
   constReg.watchProductAdditions(listening, &Listener::operator());

   Responder one("one",constReg, reg);
                 
   ModuleDescription modDesc;
   BranchDescription prod(modDesc, "int", "int", "int",0);
   reg.addProduct(prod);
   BranchDescription prod2(modDesc, "float", "float", "float",0);
   reg.addProduct(prod2);
   
   //Should be 4 products
   // 1 from the 'int' in this routine
   // 1 from 'one' responding to this call
   // 1 from the 'float'
   // 1 from 'one' responding to the original call
   CPPUNIT_ASSERT(4*2==hear);
   CPPUNIT_ASSERT(4 == reg.productList().size());
}
void  testProductRegistry:: testCircular(){
   using namespace edm;
   SignallingProductRegistry reg;
   ConstProductRegistry constReg(reg);
   
   int hear=0;
   Listener listening(hear);
   constReg.watchProductAdditions(listening);
   constReg.watchProductAdditions(listening, &Listener::operator());
   
   Responder one("one",constReg, reg);
   Responder two("two",constReg, reg);
   
   ModuleDescription modDesc;
   BranchDescription prod(modDesc, "int", "int", "int",0);
   
   reg.addProduct(prod);
   //Should be 5 products
   // 1 from the original 'add' in this routine
   // 1 from 'one' responding to this call
   // 1 from 'two' responding to 'one'
   // 1 from 'two' responding to the original call
   // 1 from 'one' responding to 'two'
   CPPUNIT_ASSERT(5*2==hear);
   CPPUNIT_ASSERT(5 == reg.productList().size());
}

void  testProductRegistry:: testProductRegistration(){
   edm::AssertHandler ah;

  const std::string config=

    "process TEST = { \n"
      "module m1 = TestPRegisterModule1{ } \n"
      "module m2 = TestPRegisterModule2{ } \n" 
      "path p = {m1,m2}\n"
      "source = TestInputSource4ProductRegistry{ }\n"
    "}\n";

  try {
   edm::EventProcessor proc(config);
  } catch(const cms::Exception& iException) {
     std::cout <<"caught "<<iException.what()<<std::endl;
     throw;
  }
}
