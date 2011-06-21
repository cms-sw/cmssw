/*----------------------------------------------------------------------

Test of the EventPrincipal class.

----------------------------------------------------------------------*/
#include "FWCore/Utilities/interface/GetPassID.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"
#include "FWCore/Utilities/interface/GlobalIdentifier.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Provenance/interface/BranchIDListHelper.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "DataFormats/Provenance/interface/WrapperInterfaceBase.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/WrapperHolder.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/GenericHandle.h"
#include "FWCore/Framework/interface/LuminosityBlockPrincipal.h"
#include "FWCore/Framework/interface/RunPrincipal.h"

#include "Cintex/Cintex.h"

#include <cppunit/extensions/HelperMacros.h>

#include <iostream>
#include <memory>
#include <string>

// This is a gross hack, to allow us to test the event
namespace edm {
   class EDProducer {
      public:
         static void commitEvent(Event& e) { e.commit_(); }
   };
}

class testGenericHandle : public CppUnit::TestFixture {
CPPUNIT_TEST_SUITE(testGenericHandle);
CPPUNIT_TEST(failgetbyLabelTest);
CPPUNIT_TEST(getbyLabelTest);
CPPUNIT_TEST(failWrongType);
CPPUNIT_TEST_SUITE_END();
public:
  void setUp(){
    ROOT::Cintex::Cintex::Enable();
  }
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
   catch (cms::Exception& x) {
      // nothing to do
   }
   catch (...) {
      CPPUNIT_ASSERT("Threw wrong kind of exception" == 0);
   }
}
void testGenericHandle::failgetbyLabelTest() {

  edm::BranchIDListHelper::clearRegistries();
  edm::EventID id;
  edm::Timestamp time;
  std::string uuid = edm::createGlobalIdentifier();
  edm::ProcessConfiguration pc("PROD", edm::ParameterSetID(), edm::getReleaseVersion(), edm::getPassID());
  boost::shared_ptr<edm::ProductRegistry const> preg(new edm::ProductRegistry);
  boost::shared_ptr<edm::RunAuxiliary> runAux(new edm::RunAuxiliary(id.run(), time, time));
  boost::shared_ptr<edm::RunPrincipal> rp(new edm::RunPrincipal(runAux, preg, pc));
  boost::shared_ptr<edm::LuminosityBlockAuxiliary> lumiAux(new edm::LuminosityBlockAuxiliary(rp->run(), 1, time, time));
  boost::shared_ptr<edm::LuminosityBlockPrincipal>lbp(new edm::LuminosityBlockPrincipal(lumiAux, preg, pc, rp));
  edm::EventAuxiliary eventAux(id, uuid, time, true);
  edm::EventPrincipal ep(preg, pc);
  ep.fillEventPrincipal(eventAux, lbp);
  edm::GenericHandle h("edmtest::DummyProduct");
  bool didThrow=true;
  try {
     edm::ParameterSet pset;
     pset.registerIt();
     edm::ModuleDescription modDesc(pset.id(), "Blah", "blahs");
     edm::Event event(ep, modDesc);

     std::string label("this does not exist");
     event.getByLabel(label,h);
     *h;
     didThrow=false;
  }
  catch (cms::Exception& x) {
    // nothing to do
  }
  catch (std::exception& x) {
    std::cout <<"caught std exception "<<x.what()<<std::endl;
    CPPUNIT_ASSERT("Threw std::exception!"==0);
  }
  catch (...) {
    CPPUNIT_ASSERT("Threw wrong kind of exception" == 0);
  }
  if (!didThrow) {
    CPPUNIT_ASSERT("Failed to throw required exception" == 0);
  }

}

void testGenericHandle::getbyLabelTest() {
  edm::BranchIDListHelper::clearRegistries();
  std::string processName = "PROD";

  typedef edmtest::DummyProduct DP;
  typedef edm::Wrapper<DP> WDP;

  std::auto_ptr<DP> pr(new DP);
  edm::WrapperHolder pprod(new WDP(pr), WDP::getInterface(), edm::WrapperHolder::Owned);
  std::string label("fred");
  std::string productInstanceName("Rick");

  edmtest::DummyProduct dp;
  edm::TypeID dummytype(dp);
  std::string className = dummytype.friendlyClassName();

  edm::ParameterSet dummyProcessPset;
  dummyProcessPset.registerIt();
  boost::shared_ptr<edm::ProcessConfiguration> processConfiguration(
    new edm::ProcessConfiguration());
  processConfiguration->setParameterSetID(dummyProcessPset.id());

  edm::ParameterSet pset;
  pset.registerIt();
  edm::ModuleDescription modDesc(pset.id(), "Blah", "", processConfiguration.get());

  edm::BranchDescription product(edm::InEvent,
                                 label,
                                 processName,
                                 dummytype.userClassName(),
                                 className,
                                 productInstanceName,
                                 modDesc,
                                 dummytype
                              );

  product.init();

  std::auto_ptr<edm::ProductRegistry> preg(new edm::ProductRegistry);
  preg->addProduct(product);
  preg->setFrozen();
  edm::BranchIDListHelper::updateRegistries(*preg);

  edm::ProductRegistry::ProductList const& pl = preg->productList();
  edm::BranchKey const bk(product);
  edm::ProductRegistry::ProductList::const_iterator it = pl.find(bk);

  edm::EventID col(1L, 1L, 1L);
  edm::Timestamp fakeTime;
  std::string uuid = edm::createGlobalIdentifier();
  edm::ProcessConfiguration pc("PROD", dummyProcessPset.id(), edm::getReleaseVersion(), edm::getPassID());
  boost::shared_ptr<edm::ProductRegistry const> pregc(preg.release());
  boost::shared_ptr<edm::RunAuxiliary> runAux(new edm::RunAuxiliary(col.run(), fakeTime, fakeTime));
  boost::shared_ptr<edm::RunPrincipal> rp(new edm::RunPrincipal(runAux, pregc, pc));
  boost::shared_ptr<edm::LuminosityBlockAuxiliary> lumiAux(new edm::LuminosityBlockAuxiliary(rp->run(), 1, fakeTime, fakeTime));
  boost::shared_ptr<edm::LuminosityBlockPrincipal>lbp(new edm::LuminosityBlockPrincipal(lumiAux, pregc, pc, rp));
  edm::EventAuxiliary eventAux(col, uuid, fakeTime, true);
  edm::EventPrincipal ep(pregc, pc);
  ep.fillEventPrincipal(eventAux, lbp);
  edm::BranchDescription const& branchFromRegistry = it->second;
  boost::shared_ptr<edm::Parentage> entryDescriptionPtr(new edm::Parentage);
  edm::ProductProvenance prov(branchFromRegistry.branchID(), entryDescriptionPtr);
  edm::ConstBranchDescription const desc(branchFromRegistry);
  ep.put(desc, pprod, prov);

  edm::GenericHandle h("edmtest::DummyProduct");
  try {
    edm::ParameterSet dummyProcessPset;
    dummyProcessPset.registerIt();
    boost::shared_ptr<edm::ProcessConfiguration> processConfiguration(
      new edm::ProcessConfiguration());
    processConfiguration->setParameterSetID(dummyProcessPset.id());

    edm::ParameterSet pset;
    pset.registerIt();
    edm::ModuleDescription modDesc(pset.id(), "Blah", "blahs", processConfiguration.get());
    edm::Event event(ep, modDesc);

    event.getByLabel(label, productInstanceName,h);
  }
  catch (cms::Exception& x) {
    std::cerr << x.explainSelf()<< std::endl;
    CPPUNIT_ASSERT("Threw cms::Exception unexpectedly" == 0);
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
