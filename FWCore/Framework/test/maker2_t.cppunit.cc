
#include <iostream>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Actions.h"
#include "FWCore/Framework/interface/ProductRegistry.h"
#include "FWCore/Framework/src/WorkerMaker.h"
#include "FWCore/Framework/src/Factory.h"
#include "FWCore/Framework/src/WorkerParams.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/parse.h"
#include "FWCore/ParameterSet/interface/Makers.h"

#include <cppunit/extensions/HelperMacros.h>

using namespace std;
using namespace edm;

class TestMod : public EDProducer
{
 public:
  explicit TestMod(ParameterSet const& p);

  void produce(Event& e, EventSetup const&);
};

TestMod::TestMod(ParameterSet const& p)
{ produces<int>();}

void TestMod::produce(Event& e, EventSetup const&)
{
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
  string param1 = "string module_type = \"TestMod\" \n"
    "string module_label = \"t1\" ";
  string param2 = "string module_type = \"TestMod\" \n"
    "string module_label = \"t2\" ";

  auto_ptr<Maker> f(new WorkerMaker<TestMod>);

  boost::shared_ptr<ParameterSet> p1 = makePSet(*edm::pset::parse(param1.c_str()));
  boost::shared_ptr<ParameterSet> p2 = makePSet(*edm::pset::parse(param2.c_str()));

  if(p1==0) cerr << "failed to make p1" << endl;
  if(p2==0) cerr << "failed to make p2" << endl;

  edm::ActionTable table;

  edm::ProductRegistry preg;
  edm::WorkerParams params1(*p1, preg, table, "PROD", 0, 0);
  edm::WorkerParams params2(*p1, preg, table, "PROD", 0, 0);

  auto_ptr<Worker> w1 = f->makeWorker(params1);
  auto_ptr<Worker> w2 = f->makeWorker(params2);

//  return 0;
}
