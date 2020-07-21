#include <sstream>

#include "CondTools/L1TriggerExt/plugins/L1CondDBIOVWriterExt.h"
#include "CondTools/L1TriggerExt/interface/DataWriterExt.h"

#include "CondFormats/L1TObjects/interface/L1TriggerKeyExt.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyExtRcd.h"
#include "CondFormats/L1TObjects/interface/L1TriggerKeyListExt.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListExtRcd.h"

#include "CondCore/CondDB/interface/Serialization.h"

L1CondDBIOVWriterExt::L1CondDBIOVWriterExt(const edm::ParameterSet& iConfig)
    : m_tscKey(iConfig.getParameter<std::string>("tscKey")),
      m_rsKey(iConfig.getParameter<std::string>("rsKey")),
      m_ignoreTriggerKey(iConfig.getParameter<bool>("ignoreTriggerKey")),
      m_logKeys(iConfig.getParameter<bool>("logKeys")),
      m_logTransactions(iConfig.getParameter<bool>("logTransactions")),
      m_forceUpdate(iConfig.getParameter<bool>("forceUpdate")) {
  //now do what ever initialization is needed
  typedef std::vector<edm::ParameterSet> ToSave;
  ToSave toSave = iConfig.getParameter<ToSave>("toPut");
  for (ToSave::const_iterator it = toSave.begin(); it != toSave.end(); it++) {
    std::string record = it->getParameter<std::string>("record");
    std::string type = it->getParameter<std::string>("type");
    m_recordTypes.push_back(record + "@" + type);
  }
}

L1CondDBIOVWriterExt::~L1CondDBIOVWriterExt() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

// ------------ method called to for each event  ------------
void L1CondDBIOVWriterExt::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  // Get L1TriggerKeyListExt
  L1TriggerKeyListExt keyList;
  l1t::DataWriterExt dataWriter;
  if (!dataWriter.fillLastTriggerKeyList(keyList)) {
    edm::LogError("L1-O2O") << "Problem getting last L1TriggerKeyListExt";
  }

  unsigned long long run = iEvent.id().run();

  L1TriggerKeyExt::RecordToKey recordTypeToKeyMap;

  bool triggerKeyIOVUpdated = true;

  // Start log string, convert run number into string
  std::stringstream ss;
  ss << run;
  std::string log = "KEYLOG runNumber=" + ss.str();
  bool logRecords = true;

  std::string m_Key = m_tscKey + ":" + m_rsKey;

  if (!m_ignoreTriggerKey) {
    if (!m_tscKey.empty() && !m_rsKey.empty()) {
      edm::LogVerbatim("L1-O2O") << "Object key for L1TriggerKeyExt@L1TriggerKeyExtRcd: " << m_tscKey << " : "
                                 << m_rsKey;

      // Use TSC key and L1TriggerKeyListExt to find next run's
      // L1TriggerKey token
      std::string keyToken = keyList.token(m_Key);

      // Update IOV sequence for this token with since-time = new run
      triggerKeyIOVUpdated = m_writer.updateIOV("L1TriggerKeyExtRcd", keyToken, run, m_logTransactions);

      // Read current L1TriggerKeyExt directly from ORCON using token
      L1TriggerKeyExt key;
      m_writer.readObject(keyToken, key);

      recordTypeToKeyMap = key.recordToKeyMap();

      // Replace spaces in key with ?s.  Do reverse substitution when
      // making L1TriggerKeyExt.
      std::string tmpKey = m_Key;
      replace(tmpKey.begin(), tmpKey.end(), ' ', '?');
      log += " tscKey:rsKey=" + tmpKey;
      logRecords = false;
    } else {
      // For use with Run Settings, no corresponding L1TrigerKey in
      // ORCON.

      // Get L1TriggerKeyExt from EventSetup
      ESHandle<L1TriggerKeyExt> esKey;
      iSetup.get<L1TriggerKeyExtRcd>().get(esKey);

      recordTypeToKeyMap = esKey->recordToKeyMap();
    }
  } else {
    std::vector<std::string>::const_iterator recordTypeItr = m_recordTypes.begin();
    std::vector<std::string>::const_iterator recordTypeEnd = m_recordTypes.end();

    for (; recordTypeItr != recordTypeEnd; ++recordTypeItr) {
      recordTypeToKeyMap.insert(std::make_pair(*recordTypeItr, m_Key));
    }
  }

  // If L1TriggerKeyExt IOV was already up to date, then so are all its
  // sub-records.
  bool throwException = false;

  if (triggerKeyIOVUpdated || m_forceUpdate) {
    // Loop over record@type in L1TriggerKeyExt
    L1TriggerKeyExt::RecordToKey::const_iterator itr = recordTypeToKeyMap.begin();
    L1TriggerKeyExt::RecordToKey::const_iterator end = recordTypeToKeyMap.end();

    for (; itr != end; ++itr) {
      std::string recordType = itr->first;
      std::string objectKey = itr->second;

      std::string recordName(recordType, 0, recordType.find_first_of('@'));

      if (logRecords) {
        // Replace spaces in key with ?s.  Do reverse substitution when
        // making L1TriggerKeyExt.
        std::string tmpKey = objectKey;
        replace(tmpKey.begin(), tmpKey.end(), ' ', '?');
        log += " " + recordName + "Key=" + tmpKey;
      }

      // Do nothing if object key is null.
      if (objectKey == L1TriggerKeyExt::kNullKey) {
        edm::LogVerbatim("L1-O2O") << "L1CondDBIOVWriterExt: null object key for " << recordType
                                   << "; skipping this record.";
      } else {
        // Find payload token
        edm::LogVerbatim("L1-O2O") << "Object key for " << recordType << ": " << objectKey;

        std::string payloadToken = keyList.token(recordType, objectKey);
        if (payloadToken.empty()) {
          edm::LogVerbatim("L1-O2O") << "L1CondDBIOVWriterExt: empty payload token for " + recordType + ", key " +
                                            objectKey;

          throwException = true;
        } else {
          m_writer.updateIOV(recordName, payloadToken, run, m_logTransactions);
        }
      }
    }
  }

  if (m_logKeys) {
    edm::LogVerbatim("L1-O2O") << log;
  }

  if (throwException) {
    throw cond::Exception("L1CondDBIOVWriterExt: empty payload tokens");
  }
}

// ------------ method called once each job just before starting event loop  ------------
void L1CondDBIOVWriterExt::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void L1CondDBIOVWriterExt::endJob() {}

//define this as a plug-in
//DEFINE_FWK_MODULE(L1CondDBIOVWriterExt);
