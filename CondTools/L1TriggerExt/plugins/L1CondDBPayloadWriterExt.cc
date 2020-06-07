#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondTools/L1TriggerExt/plugins/L1CondDBPayloadWriterExt.h"

#include "CondFormats/L1TObjects/interface/L1TriggerKeyExt.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyExtRcd.h"
#include "CondFormats/L1TObjects/interface/L1TriggerKeyListExt.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListExtRcd.h"

L1CondDBPayloadWriterExt::L1CondDBPayloadWriterExt(const edm::ParameterSet& iConfig)
    : m_writeL1TriggerKeyExt(iConfig.getParameter<bool>("writeL1TriggerKeyExt")),
      m_writeConfigData(iConfig.getParameter<bool>("writeConfigData")),
      m_overwriteKeys(iConfig.getParameter<bool>("overwriteKeys")),
      m_logTransactions(iConfig.getParameter<bool>("logTransactions")),
      m_newL1TriggerKeyListExt(iConfig.getParameter<bool>("newL1TriggerKeyListExt")) {
  //now do what ever initialization is needed
  key_token = esConsumes<L1TriggerKeyExt, L1TriggerKeyExtRcd>();
}

L1CondDBPayloadWriterExt::~L1CondDBPayloadWriterExt() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

// ------------ method called to for each event  ------------
void L1CondDBPayloadWriterExt::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  // Get L1TriggerKeyListExt and make a copy
  L1TriggerKeyListExt oldKeyList;

  if (!m_newL1TriggerKeyListExt) {
    if (!m_writer.fillLastTriggerKeyList(oldKeyList)) {
      edm::LogError("L1-O2O") << "Problem getting last L1TriggerKeyListExt";
    }
  }

  L1TriggerKeyListExt* keyList = nullptr;

  // Write L1TriggerKeyExt to ORCON with no IOV
  std::string token;
  L1TriggerKeyExt key;
  // Before calling writePayload(), check if TSC key is already in
  // L1TriggerKeyListExt.  writePayload() will not catch this situation in
  // the case of dummy configurations.
  bool triggerKeyOK = true;
  
  try {
    // Get L1TriggerKeyExt
    key = iSetup.get<L1TriggerKeyExtRcd>().get(key_token);
    if (!m_overwriteKeys) {
      triggerKeyOK = oldKeyList.token(key.tscKey()).empty();
    }
  } catch (l1t::DataAlreadyPresentException& ex) {
    triggerKeyOK = false;
    edm::LogVerbatim("L1-O2O") << ex.what();
  }

  if (triggerKeyOK && m_writeL1TriggerKeyExt) {
    edm::LogVerbatim("L1-O2O") << "Object key for L1TriggerKeyExtRcd@L1TriggerKeyExt: " << key.tscKey();
    token = m_writer.writePayload(iSetup, "L1TriggerKeyExtRcd@L1TriggerKeyExt");
  }

  // If L1TriggerKeyExt is invalid, then all configuration data is already in DB
  bool throwException = false;

  if (!token.empty() || !m_writeL1TriggerKeyExt) {
    // Record token in L1TriggerKeyListExt
    if (m_writeL1TriggerKeyExt) {
      keyList = new L1TriggerKeyListExt(oldKeyList);
      if (!(keyList->addKey(key.tscKey(), token, m_overwriteKeys))) {
        throw cond::Exception("L1CondDBPayloadWriter: TSC key " + key.tscKey() + " already in L1TriggerKeyListExt");
      }
    }

    if (m_writeConfigData) {
      // Loop over record@type in L1TriggerKeyExt
      L1TriggerKeyExt::RecordToKey::const_iterator it = key.recordToKeyMap().begin();
      L1TriggerKeyExt::RecordToKey::const_iterator end = key.recordToKeyMap().end();

      for (; it != end; ++it) {
        // Do nothing if object key is null.
        if (it->second == L1TriggerKeyExt::kNullKey) {
          edm::LogVerbatim("L1-O2O") << "L1CondDBPayloadWriter: null object key for " << it->first
                                     << "; skipping this record.";
        } else {
          // Check key is new before writing
          if (oldKeyList.token(it->first, it->second).empty() || m_overwriteKeys) {
            // Write data to ORCON with no IOV
            if (!oldKeyList.token(it->first, it->second).empty()) {
              edm::LogVerbatim("L1-O2O") << "*** Overwriting payload: object key for " << it->first << ": "
                                         << it->second;
            } else {
              edm::LogVerbatim("L1-O2O") << "object key for " << it->first << ": " << it->second;
            }

            try {
              token = m_writer.writePayload(iSetup, it->first);
            } catch (l1t::DataInvalidException& ex) {
              edm::LogVerbatim("L1-O2O") << ex.what() << " Skipping to next record.";

              throwException = true;

              continue;
            }

            if (!token.empty()) {
              // Record token in L1TriggerKeyListExt
              if (!keyList) {
                keyList = new L1TriggerKeyListExt(oldKeyList);
              }

              if (!(keyList->addKey(it->first, it->second, token, m_overwriteKeys))) {
                // This should never happen because of the check
                // above, but just in case....
                throw cond::Exception("L1CondDBPayloadWriter: subsystem key " + it->second + " for " + it->first +
                                      " already in L1TriggerKeyListExt");
              }
            }
          } else {
            edm::LogVerbatim("L1-O2O") << "L1CondDBPayloadWriter: object key " << it->second << " for " << it->first
                                       << " already in L1TriggerKeyListExt";
          }
        }
      }
    }
  }


  if (keyList) {
    // Write L1TriggerKeyListExt to ORCON
    m_writer.writeKeyList(keyList, 0, m_logTransactions);
  }

  if (throwException) {
    throw l1t::DataInvalidException("Payload problem found.");
  }
}

// ------------ method called once each job just before starting event loop  ------------
void L1CondDBPayloadWriterExt::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void L1CondDBPayloadWriterExt::endJob() {}

//define this as a plug-in
//DEFINE_FWK_MODULE(L1CondDBPayloadWriterExt);
