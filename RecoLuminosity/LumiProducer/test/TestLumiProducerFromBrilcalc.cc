// -*- C++ -*-
//
// Package:    RecoLuminosity/TestLumiProducerFromBrilcalc
// Class:      TestLumiProducerFromBrilcalc
//
/**\class TestLumiProducerFromBrilcalc TestLumiProducerFromBrilcalc.cc RecoLuminosity/LumiProducer/test/TestLumiProducerFromBrilcalc.cc

   Description: A simple analyzer class to test the functionality of TestLumiProducerFromBrilcalc.

   Implementation:
   Get the luminosity and prints it out.
*/
//
// Original Author:  Paul Lujan
//         Created:  Fri, 20 Mar 2020 09:32:27 GMT
//
//

// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Luminosity/interface/LumiInfo.h"

//
// class declaration
//

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from edm::one::EDAnalyzer<>
// This will improve performance in multithreaded jobs.

class TestLumiProducerFromBrilcalc : public edm::one::EDAnalyzer<> {
public:
  explicit TestLumiProducerFromBrilcalc(const edm::ParameterSet&);
  ~TestLumiProducerFromBrilcalc() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  edm::InputTag inputTag_;
  edm::EDGetTokenT<LumiInfo> lumiToken_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
TestLumiProducerFromBrilcalc::TestLumiProducerFromBrilcalc(const edm::ParameterSet& iConfig)
    : inputTag_(iConfig.getUntrackedParameter<edm::InputTag>("inputTag")), lumiToken_(consumes<LumiInfo>(inputTag_)) {}

TestLumiProducerFromBrilcalc::~TestLumiProducerFromBrilcalc() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called for each event  ------------
void TestLumiProducerFromBrilcalc::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  const LumiInfo& lumi = iEvent.get(lumiToken_);

  std::cout << "Luminosity for " << iEvent.run() << " LS " << iEvent.luminosityBlock() << " is "
            << lumi.getTotalInstLumi() << std::endl;
}

// ------------ method called once each job just before starting event loop  ------------
void TestLumiProducerFromBrilcalc::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void TestLumiProducerFromBrilcalc::endJob() {}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void TestLumiProducerFromBrilcalc::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // Allowed parameters
  edm::ParameterSetDescription desc;
  desc.addUntracked<edm::InputTag>("inputTag");
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestLumiProducerFromBrilcalc);
