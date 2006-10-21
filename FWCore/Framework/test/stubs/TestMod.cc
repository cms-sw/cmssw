
#include <iostream>

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

using namespace std;
using namespace edm;

class TestMod : public EDProducer
{
 public:
  explicit TestMod(ParameterSet const& p);

  void produce(Event& e, EventSetup const&);
};

TestMod::TestMod(ParameterSet const& p)
{
  produces<int>(); // We don't really produce anything.
  std::cerr << "TestMod instance created: " << p.getParameter<string>("@module_label")
	    << std::endl;
}

void TestMod::produce(Event&, EventSetup const&)
{
  std::cerr << "Hi" << std::endl;
}

DEFINE_FWK_MODULE(TestMod);
