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
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <fstream>

//
// class decleration
//

class DumpL1RPCBxOrConfig : public edm::EDAnalyzer {
public:
  explicit DumpL1RPCBxOrConfig(const edm::ParameterSet&);
  ~DumpL1RPCBxOrConfig() override;

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
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

{
  //now do what ever initialization is needed
}

DumpL1RPCBxOrConfig::~DumpL1RPCBxOrConfig() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void DumpL1RPCBxOrConfig::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  edm::ESHandle<L1RPCBxOrConfig> bxOrConfig;
  iSetup.get<L1RPCBxOrConfigRcd>().get(bxOrConfig);

  LogTrace("DumpL1RPCBxOrConfig") << std::endl;
  LogDebug("DumpL1RPCBxOrConfig") << "\n\n Printing L1RPCBxOrConfigRcd record\n" << std::endl;
  LogTrace("DumpL1RPCBxOrConfig") << "\nChecking BX Or settings: \n" << std::endl;

  LogTrace("DumpL1RPCBxOrConfig") << "First BX : " << bxOrConfig->getFirstBX()
                                  << ", Last BX : " << bxOrConfig->getLastBX() << std::endl;
}

// ------------ method called once each job just before starting event loop  ------------
void DumpL1RPCBxOrConfig::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void DumpL1RPCBxOrConfig::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(DumpL1RPCBxOrConfig);
