/*----------------------------------------------------------------------

Test of the EventPrincipal class.

----------------------------------------------------------------------*/
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"

#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/HistoryAppender.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/RootAutoLibraryLoader/interface/RootAutoLibraryLoader.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/GetPassID.h"
#include "FWCore/Utilities/interface/GlobalIdentifier.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"

#include <cppunit/extensions/HelperMacros.h>

#include "boost/shared_ptr.hpp"

#include <cassert>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeinfo>

//have to do this evil in order to access commit_ member function
#define private public
#include "FWCore/Framework/interface/Event.h"
#undef private

class testEventGetRefBeforePut: public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE(testEventGetRefBeforePut);
CPPUNIT_TEST(failGetProductNotRegisteredTest);
CPPUNIT_TEST(getRefTest);
CPPUNIT_TEST_SUITE_END();
public:
  void setUp(){
    edm::RootAutoLibraryLoader::enable();
  }
  void tearDown(){}
  void failGetProductNotRegisteredTest();
  void getRefTest();
private:
  edm::HistoryAppender historyAppender_;
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testEventGetRefBeforePut);

void testEventGetRefBeforePut::failGetProductNotRegisteredTest() {

  std::auto_ptr<edm::ProductRegistry> preg(new edm::ProductRegistry);
  preg->setFrozen();
  boost::shared_ptr<edm::BranchIDListHelper> branchIDListHelper(new edm::BranchIDListHelper());
  branchIDListHelper->updateRegistries(*preg);
  edm::EventID col(1L, 1L, 1L);
  std::string uuid = edm::createGlobalIdentifier();
  edm::Timestamp fakeTime;
  edm::ProcessConfiguration pc("PROD", edm::ParameterSetID(), edm::getReleaseVersion(), edm::getPassID());
  boost::shared_ptr<edm::ProductRegistry const> pregc(preg.release());
  boost::shared_ptr<edm::RunAuxiliary> runAux(new edm::RunAuxiliary(col.run(), fakeTime, fakeTime));
  boost::shared_ptr<edm::RunPrincipal> rp(new edm::RunPrincipal(runAux, pregc, pc, &historyAppender_));
  boost::shared_ptr<edm::LuminosityBlockAuxiliary> lumiAux(new edm::LuminosityBlockAuxiliary(rp->run(), 1, fakeTime, fakeTime));
  boost::shared_ptr<edm::LuminosityBlockPrincipal>lbp(new edm::LuminosityBlockPrincipal(lumiAux, pregc, pc, &historyAppender_));
  lbp->setRunPrincipal(rp);
  edm::EventAuxiliary eventAux(col, uuid, fakeTime, true);
  edm::EventPrincipal ep(pregc, branchIDListHelper, pc, &historyAppender_);
  ep.fillEventPrincipal(eventAux);
  ep.setLuminosityBlockPrincipal(lbp);
  try {
     edm::ParameterSet pset;
     pset.registerIt();
     boost::shared_ptr<edm::ProcessConfiguration> processConfiguration(
      new edm::ProcessConfiguration());
     edm::ModuleDescription modDesc(pset.id(), "Blah", "blahs", processConfiguration.get());
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
  edm::TypeWithDict dummytype(typeid(edmtest::IntProduct));
  std::string className = dummytype.friendlyClassName();

  edm::ParameterSet dummyProcessPset;
  dummyProcessPset.registerIt();
  boost::shared_ptr<edm::ProcessConfiguration> processConfiguration(
    new edm::ProcessConfiguration());
  processConfiguration->setParameterSetID(dummyProcessPset.id());

  edm::ParameterSet pset;
  pset.registerIt();

  edm::BranchDescription product(edm::InEvent,
                                 label,
                                 processName,
                                 dummytype.userClassName(),
                                 className,
                                 productInstanceName,
                                 "",
                                 pset.id(),
                                 dummytype
                              );

  product.init();

  std::auto_ptr<edm::ProductRegistry> preg(new edm::ProductRegistry);
  preg->addProduct(product);
  preg->setFrozen();
  boost::shared_ptr<edm::BranchIDListHelper> branchIDListHelper(new edm::BranchIDListHelper());
  branchIDListHelper->updateRegistries(*preg);
  edm::EventID col(1L, 1L, 1L);
  std::string uuid = edm::createGlobalIdentifier();
  edm::Timestamp fakeTime;
  boost::shared_ptr<edm::ProcessConfiguration> pcPtr(new edm::ProcessConfiguration(processName, dummyProcessPset.id(), edm::getReleaseVersion(), edm::getPassID()));
  edm::ProcessConfiguration& pc = *pcPtr;
  boost::shared_ptr<edm::ProductRegistry const> pregc(preg.release());
  boost::shared_ptr<edm::RunAuxiliary> runAux(new edm::RunAuxiliary(col.run(), fakeTime, fakeTime));
  boost::shared_ptr<edm::RunPrincipal> rp(new edm::RunPrincipal(runAux, pregc, pc, &historyAppender_));
  boost::shared_ptr<edm::LuminosityBlockAuxiliary> lumiAux(new edm::LuminosityBlockAuxiliary(rp->run(), 1, fakeTime, fakeTime));
  boost::shared_ptr<edm::LuminosityBlockPrincipal>lbp(new edm::LuminosityBlockPrincipal(lumiAux, pregc, pc, &historyAppender_));
  lbp->setRunPrincipal(rp);
  edm::EventAuxiliary eventAux(col, uuid, fakeTime, true);
  edm::EventPrincipal ep(pregc, branchIDListHelper, pc, &historyAppender_);
  ep.fillEventPrincipal(eventAux);
  ep.setLuminosityBlockPrincipal(lbp);

  edm::RefProd<edmtest::IntProduct> refToProd;
  try {
    edm::ModuleDescription modDesc("Blah", label, pcPtr.get());

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

