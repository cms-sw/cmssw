
#include <iostream>

#include "boost/shared_ptr.hpp"

#include "FWCore/Utilities/interface/GetPassID.h"
#include "FWCore/Version/interface/GetReleaseVersion.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/src/WorkerT.h"
#include "FWCore/Framework/interface/Actions.h"
#include "FWCore/Framework/interface/CurrentProcessingContext.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "FWCore/Framework/src/WorkerMaker.h"
#include "FWCore/Framework/src/WorkerParams.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <cppunit/extensions/HelperMacros.h>

using namespace edm;

class TestMod : public EDProducer
{
 public:
  explicit TestMod(ParameterSet const& p);

  void produce(Event& e, EventSetup const&);
};

TestMod::TestMod(ParameterSet const&)
{ produces<int>();}

void TestMod::produce(Event&, EventSetup const&)
{
  edm::CurrentProcessingContext const* p = currentContext();
  CPPUNIT_ASSERT( p != 0 );
  CPPUNIT_ASSERT( p->moduleDescription() != 0 );
  CPPUNIT_ASSERT( p->moduleLabel() != 0 );
}

// ----------------------------------------------
class testmaker2: public CppUnit::TestFixture
{
CPPUNIT_TEST_SUITE(testmaker2);
CPPUNIT_TEST(maker2Test);
CPPUNIT_TEST_SUITE_END();
public:
  void setUp(){}
  void tearDown(){}
  void maker2Test();
};

///registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(testmaker2);

void testmaker2::maker2Test()
//int main()
{
  std::auto_ptr<Maker> f(new WorkerMaker<TestMod>);

  ParameterSet p1;
  p1.addParameter("@module_type",std::string("TestMod") );
  p1.addParameter("@module_label",std::string("t1") );
  p1.addParameter("@module_edm_type",std::string("EDProducer") );
  p1.registerIt();

  ParameterSet p2;
  p2.addParameter("@module_type",std::string("TestMod") );
  p2.addParameter("@module_label",std::string("t2") );
  p2.addParameter("@module_edm_type",std::string("EDProducer") );
  p2.registerIt();

  edm::ActionTable table;

  edm::ProductRegistry preg;
  boost::shared_ptr<ProcessConfiguration> pc(new ProcessConfiguration("PROD", edm::ParameterSetID(), edm::getReleaseVersion(), edm::getPassID()));
  edm::WorkerParams params1(&p1, preg, pc, table);
  edm::WorkerParams params2(&p2, preg, pc, table);

  signalslot::Signal<void(const ModuleDescription&)> aSignal;
  std::unique_ptr<Worker> w1 = f->makeWorker(params1,aSignal,aSignal);
  std::unique_ptr<Worker> w2 = f->makeWorker(params2,aSignal,aSignal);

//  return 0;
}
