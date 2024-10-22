// -*- C++ -*-
//
// Package:    DumpL1RPCHsbConfig
// Class:      DumpL1RPCHsbConfig
//
/**\class DumpL1RPCHsbConfig DumpL1RPCHsbConfig.cc L1TriggerConfig/DumpL1RPCHsbConfig/src/DumpL1RPCHsbConfig.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/

// system include files
#include <memory>
#include "CondFormats/L1TObjects/interface/L1RPCHsbConfig.h"
#include "CondFormats/DataRecord/interface/L1RPCHsbConfigRcd.h"
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <fstream>

//
// class decleration
//

class DumpL1RPCHsbConfig : public edm::global::EDAnalyzer<> {
public:
  explicit DumpL1RPCHsbConfig(const edm::ParameterSet&);

private:
  void analyze(edm::StreamID, const edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------
  edm::ESGetToken<L1RPCHsbConfig, L1RPCHsbConfigRcd> getToken_;
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
DumpL1RPCHsbConfig::DumpL1RPCHsbConfig(const edm::ParameterSet& iConfig)
    : getToken_(esConsumes())

{
  //now do what ever initialization is needed
}

//
// member functions
//

// ------------ method called to for each event  ------------
void DumpL1RPCHsbConfig::analyze(edm::StreamID, const edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;

  edm::ESHandle<L1RPCHsbConfig> hsbConfig = iSetup.getHandle(getToken_);

  LogTrace("DumpL1RPCHsbConfig") << std::endl;
  LogDebug("DumpL1RPCHsbConfig") << "\n\n Printing L1RPCHsbConfigRcd record\n" << std::endl;
  LogTrace("DumpL1RPCHsbConfig") << "\nChecking HSB inputs: \n" << std::endl;

  LogTrace("DumpL1RPCHsbConfig") << " HSB0: ";
  for (int i = 0; i < hsbConfig->getMaskSize(); ++i) {
    LogTrace("DumpL1RPCHsbConfig") << " Input " << i << " " << hsbConfig->getHsbMask(0, i) << " ";
  }

  std::cout << std::endl;

  LogTrace("DumpL1RPCHsbConfig") << " HSB1: ";
  for (int i = 0; i < hsbConfig->getMaskSize(); ++i) {
    LogTrace("DumpL1RPCHsbConfig") << " Input " << i << " " << hsbConfig->getHsbMask(1, i) << " ";
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(DumpL1RPCHsbConfig);
