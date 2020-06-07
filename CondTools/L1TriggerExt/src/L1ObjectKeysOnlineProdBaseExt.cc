#include "CondTools/L1TriggerExt/interface/L1ObjectKeysOnlineProdBaseExt.h"

#include "CondTools/L1Trigger/interface/Exception.h"

#include "FWCore/Framework/interface/EventSetup.h"

L1ObjectKeysOnlineProdBaseExt::L1ObjectKeysOnlineProdBaseExt(const edm::ParameterSet& iConfig)
    : m_omdsReader(iConfig.getParameter<std::string>("onlineDB"),
                   iConfig.getParameter<std::string>("onlineAuthentication")) {
  //the following line is needed to tell the framework what
  // data is being produced

  // The subsystemLabel is used by L1TriggerKeyOnlineProdExt to identify the
  // L1TriggerKeysExt to concatenate.
  setWhatProduced(this, iConfig.getParameter<std::string>("subsystemLabel"))
    .setConsumes(L1TriggerKeyExt_token, edm::ESInputTag{"","SubsystemKeysOnly"});

  //now do what ever other initialization is needed
}

// edm::ESConsumesCollectorT<L1TriggerKeyExtRcd> L1ObjectKeysOnlineProdBaseExt::m_setWhatProduced(const edm::ParameterSet& iConfig) {
//   edm::ESConsumesCollectorT<L1TriggerKeyExtRcd> collector = setWhatProduced(this, iConfig.getParameter<std::string>("subsystemLabel"));
//   collector.setConsumes(L1TriggerKeyExt_token, edm::ESInputTag{"","SubsystemKeysOnly"});
//   return collector;
// }

L1ObjectKeysOnlineProdBaseExt::~L1ObjectKeysOnlineProdBaseExt() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

// ------------ method called to produce the data  ------------
L1ObjectKeysOnlineProdBaseExt::ReturnType L1ObjectKeysOnlineProdBaseExt::produce(const L1TriggerKeyExtRcd& iRecord) {
  // Get L1TriggerKeyExt with label "SubsystemKeysOnly".  Re-throw exception if
  // not present.
  L1TriggerKeyExt subsystemKeys;
  try {
    subsystemKeys = iRecord.get(L1TriggerKeyExt_token);
  } catch (l1t::DataAlreadyPresentException& ex) {
    throw ex;
  }

  // Copy L1TriggerKeyExt to new object.
  auto pL1TriggerKey = std::make_unique<L1TriggerKeyExt>(subsystemKeys);

  // Get object keys.
  fillObjectKeys(pL1TriggerKey.get());

  return pL1TriggerKey;
}


//define this as a plug-in
//DEFINE_FWK_EVENTSETUP_MODULE(L1ObjectKeysOnlineProdBaseExt);
