// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     TestPRegisterModule2
//
/**\class TestPRegisterModule2

 Description:

 Usage:

   \author Stefano ARGIRO, Chris Jones
   \date 19 May 2005
*/

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Provenance/interface/StableProvenance.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/FrameworkfwdMostUsed.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/TypeID.h"

#include "catch2/catch_test_macros.hpp"

#include <memory>
#include <vector>

class TestPRegisterModule2 : public edm::global::EDProducer<> {
public:
  explicit TestPRegisterModule2(edm::ParameterSet const&);
  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;
};

TestPRegisterModule2::TestPRegisterModule2(edm::ParameterSet const&) {
  produces<edmtest::DoubleProduct>();
  consumes<edmtest::StringProduct>(edm::InputTag{"m2"});
}

void TestPRegisterModule2::produce(edm::StreamID, edm::Event& event, edm::EventSetup const&) const {
  std::vector<edm::StableProvenance const*> plist;
  event.getAllStableProvenance(plist);

  std::vector<edm::StableProvenance const*>::const_iterator pd = plist.begin();

  REQUIRE(0 != plist.size());
  REQUIRE(2 == plist.size());
  REQUIRE(pd != plist.end());
  if (pd == plist.end())
    return;  // To silence Coverity
  edmtest::StringProduct stringprod;
  edm::TypeID stringID(stringprod);
  REQUIRE(stringID.friendlyClassName() == (*pd)->friendlyClassName());
  REQUIRE((*pd)->moduleLabel() == "m1");

  ++pd;
  REQUIRE(pd != plist.end());
  if (pd == plist.end())
    return;  // To silence Coverity

  edmtest::DoubleProduct dprod;
  edm::TypeID dID(dprod);
  REQUIRE(dID.friendlyClassName() == (*pd)->friendlyClassName());
  REQUIRE((*pd)->moduleLabel() == "m2");

  edm::Handle<edmtest::StringProduct> stringp;
  event.getByLabel("m2", stringp);
  REQUIRE(stringp->name_ == "m1");

  event.put(std::make_unique<edmtest::DoubleProduct>());
}

DEFINE_FWK_MODULE(TestPRegisterModule2);
