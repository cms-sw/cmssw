/**
   \file
   test for ProductRegistry

   \author Stefano ARGIRO
   \date 21 July 2005
*/


#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "FWCore/Framework/interface/ConstProductRegistry.h"
#include "FWCore/Framework/interface/EventProcessor.h"
#include "FWCore/Framework/src/SignallingProductRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/ProblemTracker.h"
#include "FWCore/RootAutoLibraryLoader/interface/RootAutoLibraryLoader.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

#include <cppunit/extensions/HelperMacros.h>

#include "boost/shared_ptr.hpp"

#include <iostream>

class testProductRegistry: public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE(testProductRegistry);

CPPUNIT_TEST(testSignal);
CPPUNIT_TEST(testWatch);
CPPUNIT_TEST_EXCEPTION(testCircular,cms::Exception);

CPPUNIT_TEST(testProductRegistration);

CPPUNIT_TEST_SUITE_END();

public:
  testProductRegistry();
  void setUp();
  void tearDown();
  void testSignal();
  void testWatch();
  void testCircular();
  void testProductRegistration();

 private:
  boost::shared_ptr<edm::BranchDescription> intBranch_;
  boost::shared_ptr<edm::BranchDescription> floatBranch_;
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testProductRegistry);

namespace {
   struct Listener {
      int* heard_;
      Listener(int& hear) : heard_(&hear) {}
      void operator()(edm::BranchDescription const&) {
         ++(*heard_);
      }
   };

   struct Responder {
      std::string name_;
      edm::ProductRegistry* reg_;
      Responder(std::string const& iName,
                edm::ConstProductRegistry& iConstReg,
                edm::ProductRegistry& iReg) : name_(iName),reg_(&iReg) {
        iConstReg.watchProductAdditions(this, &Responder::respond);
      }
      void respond(edm::BranchDescription const& iDesc) {
         edm::ParameterSet dummyProcessPset;
         dummyProcessPset.registerIt();
         boost::shared_ptr<edm::ProcessConfiguration> pc(
           new edm::ProcessConfiguration());
         pc->setParameterSetID(dummyProcessPset.id());

         edm::BranchDescription prod(iDesc.branchType(),
                                     name_,
                                     iDesc.processName(),
                                     iDesc.fullClassName(),
                                     iDesc.friendlyClassName(),
                                     iDesc.productInstanceName() + "-" + name_,
                                     "",
                                     iDesc.parameterSetID(),
                                     iDesc.unwrappedType()
                                    );
         reg_->addProduct(prod);
      }
   };
}

testProductRegistry::testProductRegistry() :
  intBranch_(),
  floatBranch_() {
}


void testProductRegistry::setUp() {
  edm::RootAutoLibraryLoader::enable();
  edm::ParameterSet dummyProcessPset;
  dummyProcessPset.registerIt();
  boost::shared_ptr<edm::ProcessConfiguration> processConfiguration(
    new edm::ProcessConfiguration());
  processConfiguration->setParameterSetID(dummyProcessPset.id());

  edm::ParameterSet pset;
  pset.registerIt();
  intBranch_.reset(new edm::BranchDescription(edm::InEvent, "label", "PROD",
                                          "int", "int", "int",
                                          "", pset.id(),
                                          edm::TypeWithDict(typeid(int))));

  floatBranch_.reset(new edm::BranchDescription(edm::InEvent, "label", "PROD",
                                            "float", "float", "float",
                                            "", pset.id(),
                                            edm::TypeWithDict(typeid(float))));

}

namespace {
  template <class T> void kill_and_clear(boost::shared_ptr<T>& p) { p.reset(); }
}

void testProductRegistry::tearDown() {
  kill_and_clear(floatBranch_);
  kill_and_clear(intBranch_);
}

void testProductRegistry:: testSignal() {
   using namespace edm;
   SignallingProductRegistry reg;

   int hear=0;
   Listener listening(hear);
   reg.productAddedSignal_.connect(listening);

   //BranchDescription prod(InEvent, "label", "PROD", "int", "int", "int", md);

   //   reg.addProduct(prod);
   reg.addProduct(*intBranch_);
   CPPUNIT_ASSERT(1 == hear);
}

void testProductRegistry:: testWatch() {
   using namespace edm;
   SignallingProductRegistry reg;
   ConstProductRegistry constReg(reg);

   int hear=0;
   Listener listening(hear);
   constReg.watchProductAdditions(listening);
   constReg.watchProductAdditions(listening, &Listener::operator());

   Responder one("one",constReg, reg);

   //BranchDescription prod(InEvent, "label", "PROD", "int", "int", "int");
   //reg.addProduct(prod);
   reg.addProduct(*intBranch_);

   //BranchDescription prod2(InEvent, "label", "PROD", "float", "float", "float");
   //   reg.addProduct(prod2);
   reg.addProduct(*floatBranch_);

   //Should be 4 products
   // 1 from the 'int' in this routine
   // 1 from 'one' responding to this call
   // 1 from the 'float'
   // 1 from 'one' responding to the original call
   CPPUNIT_ASSERT(4 * 2 == hear);
   CPPUNIT_ASSERT(4 == reg.size());
}
void testProductRegistry:: testCircular() {
   using namespace edm;
   SignallingProductRegistry reg;
   ConstProductRegistry constReg(reg);

   int hear=0;
   Listener listening(hear);
   constReg.watchProductAdditions(listening);
   constReg.watchProductAdditions(listening, &Listener::operator());

   Responder one("one", constReg, reg);
   Responder two("two", constReg, reg);

   //BranchDescription prod(InEvent, "label","PROD","int","int","int");
   //reg.addProduct(prod);
   reg.addProduct(*intBranch_);

   //Should be 5 products
   // 1 from the original 'add' in this routine
   // 1 from 'one' responding to this call
   // 1 from 'two' responding to 'one'
   // 1 from 'two' responding to the original call
   // 1 from 'one' responding to 'two'
   CPPUNIT_ASSERT(5 * 2 == hear);
   CPPUNIT_ASSERT(5 == reg.size());
}

void testProductRegistry:: testProductRegistration() {
   edm::AssertHandler ah;

  std::string configuration(
      "import FWCore.ParameterSet.Config as cms\n"
      "process = cms.Process('TEST')\n"
      "process.maxEvents = cms.untracked.PSet(\n"
      "  input = cms.untracked.int32(-1))\n"
      "process.source = cms.Source('DummySource')\n"
      "process.m1 = cms.EDProducer('TestPRegisterModule1')\n"
      "process.m2 = cms.EDProducer('TestPRegisterModule2')\n"
      "process.p = cms.Path(process.m1*process.m2)\n");
  try {
    edm::EventProcessor proc(configuration, true);
  } catch(cms::Exception const& iException) {
    std::cout << "caught " << iException.explainSelf() << std::endl;
    throw;
  }
}
