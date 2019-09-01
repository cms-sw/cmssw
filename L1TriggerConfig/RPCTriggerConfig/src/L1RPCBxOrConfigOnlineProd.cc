// -*- C++ -*-
//
// Package:    L1RPCBxOrConfigOnlineProd
// Class:      L1RPCBxOrConfigOnlineProd
//
/**\class L1RPCBxOrConfigOnlineProd L1RPCBxOrConfigOnlineProd.h L1Trigger/RPCConfigProducers/src/RPCBxOrConfigOnlineProd.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/

// system include files

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondTools/L1Trigger/interface/L1ConfigOnlineProdBase.h"

#include "CondFormats/L1TObjects/interface/L1RPCBxOrConfig.h"
#include "CondFormats/DataRecord/interface/L1RPCBxOrConfigRcd.h"

//
// class declaration
//

class L1RPCBxOrConfigOnlineProd : public L1ConfigOnlineProdBase<L1RPCBxOrConfigRcd, L1RPCBxOrConfig> {
public:
  L1RPCBxOrConfigOnlineProd(const edm::ParameterSet&);
  ~L1RPCBxOrConfigOnlineProd() override;

  std::unique_ptr<L1RPCBxOrConfig> newObject(const std::string& objectKey) override;

private:
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
L1RPCBxOrConfigOnlineProd::L1RPCBxOrConfigOnlineProd(const edm::ParameterSet& iConfig)
    : L1ConfigOnlineProdBase<L1RPCBxOrConfigRcd, L1RPCBxOrConfig>(iConfig) {
  //the following line is needed to tell the framework what
  // data is being produced

  //now do what ever other initialization is needed
}

L1RPCBxOrConfigOnlineProd::~L1RPCBxOrConfigOnlineProd() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

std::unique_ptr<L1RPCBxOrConfig> L1RPCBxOrConfigOnlineProd::newObject(const std::string& objectKey) {
  edm::LogError("L1-O2O") << "L1RPCBxOrConfig object with key " << objectKey << " not in ORCON!";
  auto pBxOrConfig = std::make_unique<L1RPCBxOrConfig>();
  pBxOrConfig->setFirstBX(0);
  pBxOrConfig->setLastBX(0);
  return pBxOrConfig;
}

//
// member functions
//

// ------------ method called to produce the data  ------------

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1RPCBxOrConfigOnlineProd);
