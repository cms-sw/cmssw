#include <memory>
#include <sstream>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/L1TObjects/interface/L1TriggerKey.h"
#include "CondFormats/L1TObjects/interface/L1TriggerKeyExt.h"
#include "CondFormats/L1TObjects/interface/L1TriggerKeyListExt.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyExtRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListExtRcd.h"

#include "CondTools/L1Trigger/interface/Exception.h"
#include "CondTools/L1TriggerExt/interface/DataWriterExt.h"

//
// class decleration
//

class L1O2OTestAnalyzerExt : public edm::one::EDAnalyzer<> {
public:
  explicit L1O2OTestAnalyzerExt(const edm::ParameterSet&);
  ~L1O2OTestAnalyzerExt() override;

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  bool m_printL1TriggerKeyExt;
  bool m_printL1TriggerKeyListExt;
  bool m_printESRecords;
  bool m_printPayloadTokens;
  std::vector<std::string> m_recordsToPrint;
  edm::ESGetToken<L1TriggerKeyExt, L1TriggerKeyExtRcd> l1TriggerKeyExtToken_;
};

L1O2OTestAnalyzerExt::L1O2OTestAnalyzerExt(const edm::ParameterSet& iConfig)
    : m_printL1TriggerKeyExt(iConfig.getParameter<bool>("printL1TriggerKeyExt")),
      m_printL1TriggerKeyListExt(iConfig.getParameter<bool>("printL1TriggerKeyListExt")),
      m_printESRecords(iConfig.getParameter<bool>("printESRecords")),
      m_printPayloadTokens(iConfig.getParameter<bool>("printPayloadTokens")),
      m_recordsToPrint(iConfig.getParameter<std::vector<std::string> >("recordsToPrint")) {
  //now do what ever initialization is needed
  l1TriggerKeyExtToken_ = esConsumes();
}

L1O2OTestAnalyzerExt::~L1O2OTestAnalyzerExt() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

// ------------ method called to for each event  ------------
void L1O2OTestAnalyzerExt::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  if (m_printL1TriggerKeyListExt) {
    //        ESHandle< L1TriggerKeyListExt > pList ;
    //        iSetup.get< L1TriggerKeyListExtRcd >().get( pList ) ;
    L1TriggerKeyListExt pList;
    l1t::DataWriterExt dataWriter;
    if (!dataWriter.fillLastTriggerKeyList(pList)) {
      edm::LogError("L1-O2O") << "Problem getting last L1TriggerKeyListExt";
    }

    edm::LogInfo("L1-O2O") << "Found " << pList.tscKeyToTokenMap().size() << " TSC keys:";

    L1TriggerKeyListExt::KeyToToken::const_iterator iTSCKey = pList.tscKeyToTokenMap().begin();
    L1TriggerKeyListExt::KeyToToken::const_iterator eTSCKey = pList.tscKeyToTokenMap().end();
    for (; iTSCKey != eTSCKey; ++iTSCKey) {
      edm::LogInfo("L1-O2O") << iTSCKey->first;
      if (m_printPayloadTokens) {
        edm::LogInfo("L1-O2O") << " " << iTSCKey->second;
      }
    }

    L1TriggerKeyListExt::RecordToKeyToToken::const_iterator iRec = pList.recordTypeToKeyToTokenMap().begin();
    L1TriggerKeyListExt::RecordToKeyToToken::const_iterator eRec = pList.recordTypeToKeyToTokenMap().end();
    for (; iRec != eRec; ++iRec) {
      const L1TriggerKeyListExt::KeyToToken& keyTokenMap = iRec->second;
      edm::LogInfo("L1-O2O") << "For record@type " << iRec->first << ", found " << keyTokenMap.size() << " keys:";

      L1TriggerKeyListExt::KeyToToken::const_iterator iKey = keyTokenMap.begin();
      L1TriggerKeyListExt::KeyToToken::const_iterator eKey = keyTokenMap.end();
      for (; iKey != eKey; ++iKey) {
        edm::LogInfo("L1-O2O") << iKey->first;
        if (m_printPayloadTokens) {
          edm::LogInfo("L1-O2O") << " " << iKey->second;
        }
      }
    }
  }

  if (m_printL1TriggerKeyExt) {
    try {
      ESHandle<L1TriggerKeyExt> pKey = iSetup.getHandle(l1TriggerKeyExtToken_);

      edm::LogInfo("L1-O2O") << "Current TSC key = " << pKey->tscKey();

      edm::LogInfo("L1-O2O") << "Current subsystem keys:";
      edm::LogInfo("L1-O2O") << "TSP0 " << pKey->subsystemKey(L1TriggerKeyExt::kuGT);

      edm::LogInfo("L1-O2O") << "Object keys:";
      const L1TriggerKeyExt::RecordToKey& recKeyMap = pKey->recordToKeyMap();
      L1TriggerKeyExt::RecordToKey::const_iterator iRec = recKeyMap.begin();
      L1TriggerKeyExt::RecordToKey::const_iterator eRec = recKeyMap.end();
      for (; iRec != eRec; ++iRec) {
        edm::LogInfo("L1-O2O") << iRec->first << " " << iRec->second;
      }
    } catch (cms::Exception& ex) {
      edm::LogInfo("L1-O2O") << "No L1TriggerKeyExt found.";
    }
  }

  if (m_printESRecords) {
    L1TriggerKeyListExt pList;
    l1t::DataWriterExt dataWriter;
    if (!dataWriter.fillLastTriggerKeyList(pList)) {
      edm::LogError("L1-O2O") << "Problem getting last L1TriggerKeyListExt";
    }

    // Start log string, convert run number into string
    unsigned long long run = iEvent.id().run();
    std::stringstream ss;
    ss << run;
    std::string log = "runNumber=" + ss.str();

    l1t::DataWriterExt writer;

    edm::LogInfo("L1-O2O") << "Run Settings keys:";

    std::vector<std::string>::const_iterator iRec = m_recordsToPrint.begin();
    std::vector<std::string>::const_iterator iEnd = m_recordsToPrint.end();
    for (; iRec != iEnd; ++iRec) {
      std::string payloadToken = writer.payloadToken(*iRec, iEvent.id().run());
      std::string key;

      if (*iRec == "L1TriggerKeyExtRcd") {
        key = pList.tscKey(payloadToken);
      } else {
        key = pList.objectKey(*iRec, payloadToken);
      }

      edm::LogInfo("L1-O2O") << *iRec << " " << key;
      if (m_printPayloadTokens) {
        edm::LogInfo("L1-O2O") << " " << payloadToken;
      }

      // Replace spaces in key with ?s.  Do reverse substitution when
      // making L1TriggerKeyExt.
      replace(key.begin(), key.end(), ' ', '?');
      log += " " + *iRec + "Key=" + key;
    }

    edm::LogInfo("L1-O2O") << log;
  }
}

// ------------ method called once each job just before starting event loop  ------------
void L1O2OTestAnalyzerExt::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void L1O2OTestAnalyzerExt::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(L1O2OTestAnalyzerExt);
