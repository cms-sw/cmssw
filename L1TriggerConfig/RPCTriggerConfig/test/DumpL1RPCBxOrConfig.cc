// -*- C++ -*-
//
// Package:    DumpL1RPCBxOrConfig
// Class:      DumpL1RPCBxOrConfig
//
/**\class DumpL1RPCBxOrConfig DumpL1RPCBxOrConfig.cc L1TriggerConfig/DumpL1RPCBxOrConfig/src/DumpL1RPCBxOrConfig.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/

// system include files
#include <memory>
#include "CondFormats/L1TObjects/interface/L1RPCBxOrConfig.h"
#include "CondFormats/DataRecord/interface/L1RPCBxOrConfigRcd.h"
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

class DumpL1RPCBxOrConfig : public edm::global::EDAnalyzer<> {
public:
  explicit DumpL1RPCBxOrConfig(const edm::ParameterSet&);

private:
  void analyze(edm::StreamID, const edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------
  edm::ESGetToken<L1RPCBxOrConfig, L1RPCBxOrConfigRcd> getToken_;
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
DumpL1RPCBxOrConfig::DumpL1RPCBxOrConfig(const edm::ParameterSet& iConfig)
    : getToken_(esConsumes())

{
  //now do what ever initialization is needed
}

//
// member functions
//

// ------------ method called to for each event  ------------
void DumpL1RPCBxOrConfig::analyze(edm::StreamID, const edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;

  L1RPCBxOrConfig const& bxOrConfig = iSetup.getData(getToken_);

  LogTrace("DumpL1RPCBxOrConfig") << std::endl;
  LogDebug("DumpL1RPCBxOrConfig") << "\n\n Printing L1RPCBxOrConfigRcd record\n" << std::endl;
  LogTrace("DumpL1RPCBxOrConfig") << "\nChecking BX Or settings: \n" << std::endl;

  LogTrace("DumpL1RPCBxOrConfig") << "First BX : " << bxOrConfig.getFirstBX()
                                  << ", Last BX : " << bxOrConfig.getLastBX() << std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(DumpL1RPCBxOrConfig);
