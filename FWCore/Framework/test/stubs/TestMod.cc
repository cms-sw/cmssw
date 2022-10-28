
#include <iostream>

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

using namespace edm;

class TestMod : public global::EDProducer<> {
public:
  explicit TestMod(ParameterSet const& p);

  void produce(StreamID, Event& e, EventSetup const&) const final;
};

TestMod::TestMod(ParameterSet const& p) {
  produces<int>();  // We don't really produce anything.
  //std::cerr << "TestMod instance created: " << p.getParameter<std::string>("@module_label")
  //    << std::endl;
}

void TestMod::produce(StreamID, Event&, EventSetup const&) const {
  //std::cerr << "Hi" << std::endl;
}

DEFINE_FWK_MODULE(TestMod);
