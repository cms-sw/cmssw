// -*- C++ -*-
//
// Package:    L1RPCConeDefinitionOnlineProd
// Class:      L1RPCConeDefinitionOnlineProd
//
/**\class L1RPCConeDefinitionOnlineProd L1RPCConeDefinitionOnlineProd.h L1Trigger/L1RPCConeDefinitionProducers/src/L1RPCConeDefinitionOnlineProd.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Werner Man-Li Sun
//         Created:  Wed Apr  1 20:23:43 CEST 2009
// $Id$
//
//

// system include files

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondTools/L1Trigger/interface/L1ConfigOnlineProdBase.h"

#include "CondFormats/L1TObjects/interface/L1RPCConeDefinition.h"
#include "CondFormats/DataRecord/interface/L1RPCConeDefinitionRcd.h"

//
// class declaration
//

class L1RPCConeDefinitionOnlineProd : public L1ConfigOnlineProdBase<L1RPCConeDefinitionRcd, L1RPCConeDefinition> {
public:
  L1RPCConeDefinitionOnlineProd(const edm::ParameterSet&);
  ~L1RPCConeDefinitionOnlineProd() override;

  std::unique_ptr<L1RPCConeDefinition> newObject(const std::string& objectKey) override;

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
L1RPCConeDefinitionOnlineProd::L1RPCConeDefinitionOnlineProd(const edm::ParameterSet& iConfig)
    : L1ConfigOnlineProdBase<L1RPCConeDefinitionRcd, L1RPCConeDefinition>(iConfig) {
  //the following line is needed to tell the framework what
  // data is being produced

  //now do what ever other initialization is needed
}

L1RPCConeDefinitionOnlineProd::~L1RPCConeDefinitionOnlineProd() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

std::unique_ptr<L1RPCConeDefinition> L1RPCConeDefinitionOnlineProd::newObject(const std::string& objectKey) {
  edm::LogError("L1-O2O") << "L1RPCConeDefinition object with key " << objectKey << " not in ORCON!";

  return std::unique_ptr<L1RPCConeDefinition>();
}

//
// member functions
//

// ------------ method called to produce the data  ------------

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(L1RPCConeDefinitionOnlineProd);
