#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CondFormats/L1TObjects/interface/L1TriggerKeyExt.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyExtRcd.h"
#include "CondFormats/L1TObjects/interface/L1TriggerKeyListExt.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListExtRcd.h"

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "CondTools/L1TriggerExt/interface/DataWriterExt.h"

using DataWriterExtPtr = std::unique_ptr<l1t::DataWriterExt>;
using RecordToWriterMap = std::map<std::string, DataWriterExtPtr>;

class L1CondDBPayloadWriterExt : public edm::one::EDAnalyzer<> {
public:
  explicit L1CondDBPayloadWriterExt(const edm::ParameterSet&);
  ~L1CondDBPayloadWriterExt() override;

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  RecordToWriterMap m_rcdToWriterMap;
  // std::string m_tag ; // tag is known by PoolDBOutputService

  // set to false to write config data without valid TSC key
  bool m_writeL1TriggerKeyExt;

  // set to false to write config data only
  bool m_writeConfigData;

  // substitute new payload tokens for existing keys in L1TriggerKeyListExt
  bool m_overwriteKeys;

  bool m_logTransactions;

  // if true, do not retrieve L1TriggerKeyListExt from EventSetup
  bool m_newL1TriggerKeyListExt;

  // Token to access L1TriggerKeyExt data in the event setup
  edm::ESGetToken<L1TriggerKeyExt, L1TriggerKeyExtRcd> theL1TriggerKeyExtToken_;
};

L1CondDBPayloadWriterExt::L1CondDBPayloadWriterExt(const edm::ParameterSet& iConfig)
    : m_writeL1TriggerKeyExt(iConfig.getParameter<bool>("writeL1TriggerKeyExt")),
      m_writeConfigData(iConfig.getParameter<bool>("writeConfigData")),
      m_overwriteKeys(iConfig.getParameter<bool>("overwriteKeys")),
      m_logTransactions(iConfig.getParameter<bool>("logTransactions")),
      m_newL1TriggerKeyListExt(iConfig.getParameter<bool>("newL1TriggerKeyListExt")),
      theL1TriggerKeyExtToken_(esConsumes()) {
  auto cc = consumesCollector();
  for (const auto& sysWriter : iConfig.getParameter<std::vector<std::string>>("sysWriters")) {
    //construct writer
    DataWriterExtPtr writer = std::make_unique<l1t::DataWriterExt>(sysWriter);
    writer->getWriter()->setToken(cc);
    m_rcdToWriterMap[sysWriter] = std::move(writer);  //the sysWriter holds info in 'rcd@prod' format
  }
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
  l1t::DataWriterExt& m_writer = *m_rcdToWriterMap.at("L1TriggerKeyExtRcd@L1TriggerKeyExt");

  if (not(m_newL1TriggerKeyListExt or m_writer.fillLastTriggerKeyList(oldKeyList)))
    edm::LogError("L1-O2O") << "Problem getting last L1TriggerKeyListExt";

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
    key = iSetup.getData(theL1TriggerKeyExtToken_);
    if (!m_overwriteKeys) {
      triggerKeyOK = oldKeyList.token(key.tscKey()).empty();
    }
  } catch (l1t::DataAlreadyPresentException& ex) {
    triggerKeyOK = false;
    edm::LogVerbatim("L1-O2O") << ex.what();
  }

  if (triggerKeyOK and m_writeL1TriggerKeyExt) {
    edm::LogVerbatim("L1-O2O") << "Object key for L1TriggerKeyExtRcd@L1TriggerKeyExt: " << key.tscKey()
                               << " (about to run writePayload)";
    token = m_writer.writePayload(iSetup);
  }

  // If L1TriggerKeyExt is invalid (empty), then all configuration data is already in DB
  // m_writeL1TriggerKeyExt the naming is misleading,
  // the bool is used to say to the module whether it runs or not to update a L1TriggerKeyExtRcd
  // (so if no payload for L1TriggerKeyExtRcd AND you run for updating L1TriggerKeyExtRcd ==> you have nothing to update)
  if (token.empty() and m_writeL1TriggerKeyExt) {
    edm::LogInfo("L1CondDBPayloadWriterExt::analyze") << " token = " << token;
    return;
  }

  // Record token in L1TriggerKeyListExt
  if (m_writeL1TriggerKeyExt) {
    keyList = new L1TriggerKeyListExt(oldKeyList);
    if (not keyList->addKey(key.tscKey(), token, m_overwriteKeys))
      throw cond::Exception("L1CondDBPayloadWriter: TSC key " + key.tscKey() + " already in L1TriggerKeyListExt");
  }

  if (not m_writeConfigData) {
    // Write L1TriggerKeyListExt to ORCON
    if (keyList)
      m_writer.writeKeyList(keyList, 0, m_logTransactions);
    return;
  }

  // Loop over record@type in L1TriggerKeyExt
  // (as before make writers, try to write payload and if needed handle exceptions)

  // this is not needed maps have implemented begin and end methods for their iterators
  // L1TriggerKeyExt::RecordToKey::const_iterator it = key.recordToKeyMap().begin();
  // L1TriggerKeyExt::RecordToKey::const_iterator end = key.recordToKeyMap().end();
  // for (; it != end; ++it) {

  bool throwException = false;
  for (const auto& it : key.recordToKeyMap()) {
    // If there isn't any WriterProxyT constructed for this rcd, continue
    // (the missing rcds are left out for a reason - those are static that throw exceptions that cannot be handled in 12_3)
    if (m_rcdToWriterMap.find(it.first) == m_rcdToWriterMap.end())
      continue;

    // Do nothing if object key is null.
    // (Panos) this might not be working as the "empty" keys are L1TriggerKeyExt::kEmptyKey (std::string(""))
    if (it.second == L1TriggerKeyExt::kNullKey) {
      edm::LogVerbatim("L1-O2O") << "L1CondDBPayloadWriter: null object key for " << it.first
                                 << "; skipping this record.";
      continue;
    }

    // Check key is new before writing
    if (oldKeyList.token(it.first, it.second).empty() || m_overwriteKeys) {
      // Write data to ORCON with no IOV
      if (!oldKeyList.token(it.first, it.second).empty()) {
        edm::LogVerbatim("L1-O2O") << "*** Overwriting payload: object key for " << it.first << ": " << it.second;
      } else {
        edm::LogVerbatim("L1-O2O") << "object key for " << it.first << ": " << it.second;
      }

      try {
        edm::LogVerbatim("L1-O2O") << "about to run writePayload for " << it.first;
        token = m_rcdToWriterMap.at(it.first)->writePayload(iSetup);
      } catch (l1t::DataInvalidException& ex) {
        edm::LogVerbatim("L1-O2O") << ex.what() << " Skipping to next record.";
        throwException = true;
        continue;
      }

      if (!token.empty()) {
        // Record token in L1TriggerKeyListExt
        if (!keyList)
          keyList = new L1TriggerKeyListExt(oldKeyList);
        // The following should never happen because of the check
        // above, but just in case....
        if (!(keyList->addKey(it.first, it.second, token, m_overwriteKeys)))
          throw cond::Exception("L1CondDBPayloadWriter")
              << "subsystem key " << it.second << " for " << it.first << " already in L1TriggerKeyListExt";
      }

    } else
      edm::LogVerbatim("L1-O2O") << "L1CondDBPayloadWriter: object key " << it.second << " for " << it.first
                                 << " already in L1TriggerKeyListExt";

  }  // for rcds from keys

  if (keyList)  // Write L1TriggerKeyListExt to ORCON
    m_writer.writeKeyList(keyList, 0, m_logTransactions);

  if (throwException)
    throw l1t::DataInvalidException("Payload problem found.");
}

// ------------ method called once each job just before starting event loop  ------------
void L1CondDBPayloadWriterExt::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void L1CondDBPayloadWriterExt::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(L1CondDBPayloadWriterExt);
