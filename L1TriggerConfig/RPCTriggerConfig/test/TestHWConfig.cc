// -*- C++ -*-
//
// Package:    TestHWConfig
// Class:      TestHWConfig
//
/**\class TestHWConfig TestHWConfig.cc L1TriggerConfig/TestHWConfig/src/TestHWConfig.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Tomasz Maciej Frueboes
//         Created:  Wed Apr  9 14:03:40 CEST 2008
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/DataRecord/interface/L1RPCHwConfigRcd.h"
#include "CondFormats/RPCObjects/interface/L1RPCHwConfig.h"

//
// class decleration
//

class TestHWConfig : public edm::global::EDAnalyzer<> {
public:
  explicit TestHWConfig(const edm::ParameterSet&);

private:
  void analyze(edm::StreamID, const edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------
  edm::ESGetToken<L1RPCHwConfig, L1RPCHwConfigRcd> getToken_;
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
TestHWConfig::TestHWConfig(const edm::ParameterSet& iConfig)
    : getToken_(esConsumes())

{
  //now do what ever initialization is needed
}

//
// member functions
//

// ------------ method called to for each event  ------------
void TestHWConfig::analyze(edm::StreamID, const edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;
  L1RPCHwConfig const& hwConfig = iSetup.getData(getToken_);

  std::cout << "Checking crates " << std::endl;

  for (int crate = 0; crate < 12; ++crate) {
    std::set<int> enabledTowers;

    for (int tw = -16; tw < 17; ++tw) {
      if (hwConfig.isActive(tw, crate, 0))
        enabledTowers.insert(tw);
    }

    if (!enabledTowers.empty()) {
      std::cout << "Crate " << crate << ", active towers:";

      std::set<int>::iterator it;
      for (it = enabledTowers.begin(); it != enabledTowers.end(); ++it) {
        std::cout << " " << *it;
      }
      std::cout << std::endl;

    }  // printout

  }  // crate iteration ends

  //std::cout << "First BX: "<<hwConfig->getFirstBX()<<", last BX: "<<hwConfig->getLastBX()<<std::endl;

  std::cout << " Done " << hwConfig.size() << std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(TestHWConfig);
