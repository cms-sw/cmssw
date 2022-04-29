
#include <iostream>

#include <memory>

#include "FWCore/Utilities/interface/GetPassID.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/maker/WorkerT.h"
#include "FWCore/Framework/interface/PreallocationConfiguration.h"
#include "FWCore/Framework/interface/ExceptionActions.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Framework/interface/maker/WorkerMaker.h"
#include "FWCore/Framework/interface/maker/MakeModuleParams.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "cppunit/extensions/HelperMacros.h"

using namespace edm;

class TestMod : public global::EDProducer<> {
public:
  explicit TestMod(ParameterSet const& p);

  void produce(StreamID, Event& e, EventSetup const&) const override;
};

TestMod::TestMod(ParameterSet const&) { produces<int>(); }

void TestMod::produce(StreamID, Event&, EventSetup const&) const {}

// ----------------------------------------------
class testmaker2 : public CppUnit::TestFixture {
  CPPUNIT_TEST_SUITE(testmaker2);
  CPPUNIT_TEST(maker2Test);
  CPPUNIT_TEST_SUITE_END();

public:
  void setUp() {}
  void tearDown() {}
  void maker2Test();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testmaker2);

void testmaker2::maker2Test()
//int main()
{
  std::unique_ptr<Maker> f = std::make_unique<WorkerMaker<TestMod>>();

  ParameterSet p1;
  p1.addParameter("@module_type", std::string("TestMod"));
  p1.addParameter("@module_label", std::string("t1"));
  p1.addParameter("@module_edm_type", std::string("EDProducer"));
  p1.registerIt();

  ParameterSet p2;
  p2.addParameter("@module_type", std::string("TestMod"));
  p2.addParameter("@module_label", std::string("t2"));
  p2.addParameter("@module_edm_type", std::string("EDProducer"));
  p2.registerIt();

  edm::ExceptionToActionTable table;

  edm::ProductRegistry preg;
  edm::PreallocationConfiguration prealloc;
  auto pc =
      std::make_shared<ProcessConfiguration>("PROD", edm::ParameterSetID(), edm::getReleaseVersion(), edm::getPassID());
  edm::MakeModuleParams params1(&p1, preg, &prealloc, pc);
  edm::MakeModuleParams params2(&p2, preg, &prealloc, pc);

  signalslot::Signal<void(const ModuleDescription&)> aSignal;
  auto m1 = f->makeModule(params1, aSignal, aSignal);
  std::unique_ptr<Worker> w1 = m1->makeWorker(&table);
  auto m2 = f->makeModule(params2, aSignal, aSignal);
  std::unique_ptr<Worker> w2 = m2->makeWorker(&table);

  //  return 0;
}
