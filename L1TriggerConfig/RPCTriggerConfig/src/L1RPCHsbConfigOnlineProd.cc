// -*- C++ -*-
//
// Package:    L1RPCHsbConfigOnlineProd
// Class:      L1RPCHsbConfigOnlineProd
//
/**\class L1RPCHsbConfigOnlineProd L1RPCHsbConfigOnlineProd.h L1Trigger/RPCConfigProducers/src/RPCHsbConfigOnlineProd.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/

// system include files

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondTools/L1Trigger/interface/L1ConfigOnlineProdBase.h"

#include "CondFormats/L1TObjects/interface/L1RPCHsbConfig.h"
#include "CondFormats/DataRecord/interface/L1RPCHsbConfigRcd.h"

//
// class declaration
//

class L1RPCHsbConfigOnlineProd : public L1ConfigOnlineProdBase<L1RPCHsbConfigRcd, L1RPCHsbConfig> {
public:
  L1RPCHsbConfigOnlineProd(const edm::ParameterSet&);
  ~L1RPCHsbConfigOnlineProd() override;

  std::unique_ptr<L1RPCHsbConfig> newObject(const std::string& objectKey) override;

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
L1RPCHsbConfigOnlineProd::L1RPCHsbConfigOnlineProd(const edm::ParameterSet& iConfig)
    : L1ConfigOnlineProdBase<L1RPCHsbConfigRcd, L1RPCHsbConfig>(iConfig) {
  //the following line is needed to tell the framework what
  // data is being produced

  //now do what ever other initialization is needed
}

L1RPCHsbConfigOnlineProd::~L1RPCHsbConfigOnlineProd() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

std::unique_ptr<L1RPCHsbConfig> L1RPCHsbConfigOnlineProd::newObject(const std::string& objectKey) {
  edm::LogError("L1-O2O") << "L1RPCHsbConfig object with key " << objectKey << " not in ORCON!";
  auto pHsbConfig = std::make_unique<L1RPCHsbConfig>();
  std::vector<int> hsbconf;
  int mask = 3;
  // XX was: i<9, corrected
  hsbconf.reserve(8);

for (int i = 0; i < 8; i++)
    hsbconf.push_back(mask);
  pHsbConfig->setHsbMask(0, hsbconf);
  pHsbConfig->setHsbMask(1, hsbconf);
  return pHsbConfig;
}

//
// member functions
//

// ------------ method called to produce the data  ------------

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1RPCHsbConfigOnlineProd);
