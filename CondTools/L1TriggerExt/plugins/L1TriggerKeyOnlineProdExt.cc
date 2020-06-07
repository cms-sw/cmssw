#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondTools/L1TriggerExt/plugins/L1TriggerKeyOnlineProdExt.h"

#include "CondTools/L1Trigger/interface/Exception.h"

#include "FWCore/Framework/interface/EventSetup.h"

L1TriggerKeyOnlineProdExt::L1TriggerKeyOnlineProdExt(const edm::ParameterSet& iConfig)  {
  //the following line is needed to tell the framework what
  // data is being produced
  auto cc = setWhatProduced(this);
  
  for(auto const& label : iConfig.getParameter<std::vector<std::string> >("subsystemLabels"))  {
    m_subsystemTokens.emplace_back(cc.consumesFrom<L1TriggerKeyExt, L1TriggerKeyExtRcd>(edm::ESInputTag{"",label}));
  
  //now do what ever other initialization is needed
  }

  cc.setConsumes(L1TriggerKeyExt_token, edm::ESInputTag{"","SubsystemKeysOnly"});
}

L1TriggerKeyOnlineProdExt::~L1TriggerKeyOnlineProdExt() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to produce the data  ------------
L1TriggerKeyOnlineProdExt::ReturnType L1TriggerKeyOnlineProdExt::produce(const L1TriggerKeyExtRcd& iRecord) {
  // Start with "SubsystemKeysOnly"
  L1TriggerKeyExt subsystemKeys;
  try {
    subsystemKeys = iRecord.get(L1TriggerKeyExt_token);
  } catch (l1t::DataAlreadyPresentException& ex) {
    throw ex;
  }

  auto pL1TriggerKey = std::make_unique<L1TriggerKeyExt>(subsystemKeys);

  // Collate object keys
  for(auto const& token: m_subsystemTokens) {
    pL1TriggerKey->add(iRecord.get(token).recordToKeyMap());
  }

  return pL1TriggerKey;
}

//define this as a plug-in
//DEFINE_FWK_EVENTSETUP_MODULE(L1TriggerKeyOnlineProdExt);
