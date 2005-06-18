
#include <iostream>

#include "FWCore/CoreFramework/interface/EDProducer.h"
#include "FWCore/CoreFramework/interface/Event.h"
#include "FWCore/CoreFramework/interface/Handle.h"
#include "FWCore/CoreFramework/interface/MakerMacros.h"

#include "FWCore/CoreFramework/interface/EventSetup.h"
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
  std::cerr << "TestMod instance created: " << p.getParameter<string>("module_label")
	    << std::endl;
}

void TestMod::produce(Event& e, EventSetup const&)
{
  std::cerr << "Hi" << std::endl;
}

DEFINE_FWK_MODULE(TestMod)
