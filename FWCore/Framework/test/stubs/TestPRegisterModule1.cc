// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     TestPRegisterModule1
//
/**\class TestPRegisterModule1

 Description:

 Usage:

   \author Stefano ARGIRO, Chris Jones
   \date 19 May 2005
*/

#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <memory>
#include <string>

class TestPRegisterModule1 : public edm::global::EDProducer<> {
public:
  explicit TestPRegisterModule1(edm::ParameterSet const&);
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

private:
  edm::ParameterSet pset_;
};

TestPRegisterModule1::TestPRegisterModule1(edm::ParameterSet const& pset) : pset_(pset) {
  produces<edmtest::StringProduct>();
}

void TestPRegisterModule1::produce(edm::StreamID, edm::Event& event, edm::EventSetup const&) const {
  std::string myname = pset_.getParameter<std::string>("@module_label");
  event.put(std::make_unique<edmtest::StringProduct>(myname));
}

DEFINE_FWK_MODULE(TestPRegisterModule1);
