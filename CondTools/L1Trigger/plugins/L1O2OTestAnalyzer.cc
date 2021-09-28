// -*- C++ -*-
//
// Package:    L1O2OTestAnalyzer
// Class:      L1O2OTestAnalyzer
//
/**\class L1O2OTestAnalyzer L1O2OTestAnalyzer.cc CondTools/L1O2OTestAnalyzer/src/L1O2OTestAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Werner Man-Li Sun
//         Created:  Thu Nov  6 23:00:43 CET 2008
//
//

// system include files
#include <memory>
#include <sstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/L1TObjects/interface/L1TriggerKey.h"
#include "CondFormats/L1TObjects/interface/L1TriggerKeyList.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"
#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"

#include "CondTools/L1Trigger/interface/Exception.h"
#include "CondTools/L1Trigger/interface/DataWriter.h"

//
// class decleration
//

class L1O2OTestAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit L1O2OTestAnalyzer(const edm::ParameterSet&);
  ~L1O2OTestAnalyzer() override;

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  // ----------member data ---------------------------
  bool m_printL1TriggerKey;
  bool m_printL1TriggerKeyList;
  bool m_printESRecords;
  bool m_printPayloadTokens;
  std::vector<std::string> m_recordsToPrint;
  edm::ESGetToken<L1TriggerKey, L1TriggerKeyRcd> l1TriggerKeyToken_;
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
L1O2OTestAnalyzer::L1O2OTestAnalyzer(const edm::ParameterSet& iConfig)
    : m_printL1TriggerKey(iConfig.getParameter<bool>("printL1TriggerKey")),
      m_printL1TriggerKeyList(iConfig.getParameter<bool>("printL1TriggerKeyList")),
      m_printESRecords(iConfig.getParameter<bool>("printESRecords")),
      m_printPayloadTokens(iConfig.getParameter<bool>("printPayloadTokens")),
      m_recordsToPrint(iConfig.getParameter<std::vector<std::string> >("recordsToPrint")) {
  //now do what ever initialization is needed
  l1TriggerKeyToken_ = esConsumes();
}

L1O2OTestAnalyzer::~L1O2OTestAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void L1O2OTestAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  if (m_printL1TriggerKeyList) {
    //        ESHandle< L1TriggerKeyList > pList ;
    //        iSetup.get< L1TriggerKeyListRcd >().get( pList ) ;
    L1TriggerKeyList pList;
    l1t::DataWriter dataWriter;
    if (!dataWriter.fillLastTriggerKeyList(pList)) {
      edm::LogError("L1-O2O") << "Problem getting last L1TriggerKeyList";
    }

    edm::LogInfo("L1-O2O") << "Found " << pList.tscKeyToTokenMap().size() << " TSC keys:" << std::endl;

    L1TriggerKeyList::KeyToToken::const_iterator iTSCKey = pList.tscKeyToTokenMap().begin();
    L1TriggerKeyList::KeyToToken::const_iterator eTSCKey = pList.tscKeyToTokenMap().end();
    for (; iTSCKey != eTSCKey; ++iTSCKey) {
      edm::LogInfo("L1-O2O") << iTSCKey->first;
      if (m_printPayloadTokens) {
        edm::LogInfo("L1-O2O") << " " << iTSCKey->second;
      }
      edm::LogInfo("L1-O2O") << std::endl;
    }
    edm::LogInfo("L1-O2O") << std::endl;

    L1TriggerKeyList::RecordToKeyToToken::const_iterator iRec = pList.recordTypeToKeyToTokenMap().begin();
    L1TriggerKeyList::RecordToKeyToToken::const_iterator eRec = pList.recordTypeToKeyToTokenMap().end();
    for (; iRec != eRec; ++iRec) {
      const L1TriggerKeyList::KeyToToken& keyTokenMap = iRec->second;
      edm::LogInfo("L1-O2O") << "For record@type " << iRec->first << ", found " << keyTokenMap.size()
                             << " keys:" << std::endl;

      L1TriggerKeyList::KeyToToken::const_iterator iKey = keyTokenMap.begin();
      L1TriggerKeyList::KeyToToken::const_iterator eKey = keyTokenMap.end();
      for (; iKey != eKey; ++iKey) {
        edm::LogInfo("L1-O2O") << iKey->first;
        if (m_printPayloadTokens) {
          edm::LogInfo("L1-O2O") << " " << iKey->second;
        }
        edm::LogInfo("L1-O2O") << std::endl;
      }
      edm::LogInfo("L1-O2O") << std::endl;
    }
  }

  if (m_printL1TriggerKey) {
    try {
      auto pKey = iSetup.getHandle(l1TriggerKeyToken_);

      edm::LogInfo("L1-O2O") << std::endl;
      edm::LogInfo("L1-O2O") << "Current TSC key = " << pKey->tscKey() << std::endl << std::endl;

      edm::LogInfo("L1-O2O") << "Current subsystem keys:" << std::endl;
      edm::LogInfo("L1-O2O") << "CSCTF " << pKey->subsystemKey(L1TriggerKey::kCSCTF) << std::endl;
      edm::LogInfo("L1-O2O") << "DTTF " << pKey->subsystemKey(L1TriggerKey::kDTTF) << std::endl;
      edm::LogInfo("L1-O2O") << "RPC " << pKey->subsystemKey(L1TriggerKey::kRPC) << std::endl;
      edm::LogInfo("L1-O2O") << "GMT " << pKey->subsystemKey(L1TriggerKey::kGMT) << std::endl;
      edm::LogInfo("L1-O2O") << "RCT " << pKey->subsystemKey(L1TriggerKey::kRCT) << std::endl;
      edm::LogInfo("L1-O2O") << "GCT " << pKey->subsystemKey(L1TriggerKey::kGCT) << std::endl;
      edm::LogInfo("L1-O2O") << "GT " << pKey->subsystemKey(L1TriggerKey::kGT) << std::endl;
      edm::LogInfo("L1-O2O") << "TSP0 " << pKey->subsystemKey(L1TriggerKey::kTSP0) << std::endl << std::endl;

      edm::LogInfo("L1-O2O") << "Object keys:" << std::endl;
      const L1TriggerKey::RecordToKey& recKeyMap = pKey->recordToKeyMap();
      L1TriggerKey::RecordToKey::const_iterator iRec = recKeyMap.begin();
      L1TriggerKey::RecordToKey::const_iterator eRec = recKeyMap.end();
      for (; iRec != eRec; ++iRec) {
        edm::LogInfo("L1-O2O") << iRec->first << " " << iRec->second << std::endl;
      }
    } catch (cms::Exception& ex) {
      edm::LogError("L1-O2O") << "No L1TriggerKey found." << std::endl;
    }
  }

  if (m_printESRecords) {
    //        ESHandle< L1TriggerKeyList > pList ;
    //        iSetup.get< L1TriggerKeyListRcd >().get( pList ) ;

    L1TriggerKeyList pList;
    l1t::DataWriter dataWriter;
    if (!dataWriter.fillLastTriggerKeyList(pList)) {
      edm::LogError("L1-O2O") << "Problem getting last L1TriggerKeyList";
    }

    // Start log string, convert run number into string
    unsigned long long run = iEvent.id().run();
    std::stringstream ss;
    ss << run;
    std::string log = "runNumber=" + ss.str();

    l1t::DataWriter writer;

    edm::LogInfo("L1-O2O") << std::endl << "Run Settings keys:" << std::endl;

    std::vector<std::string>::const_iterator iRec = m_recordsToPrint.begin();
    std::vector<std::string>::const_iterator iEnd = m_recordsToPrint.end();
    for (; iRec != iEnd; ++iRec) {
      std::string payloadToken = writer.payloadToken(*iRec, iEvent.id().run());
      std::string key;

      if (*iRec == "L1TriggerKeyRcd") {
        key = pList.tscKey(payloadToken);
      } else {
        key = pList.objectKey(*iRec, payloadToken);
      }

      edm::LogInfo("L1-O2O") << *iRec << " " << key;
      if (m_printPayloadTokens) {
        edm::LogInfo("L1-O2O") << " " << payloadToken;
      }
      edm::LogInfo("L1-O2O") << std::endl;

      // Replace spaces in key with ?s.  Do reverse substitution when
      // making L1TriggerKey.
      replace(key.begin(), key.end(), ' ', '?');
      log += " " + *iRec + "Key=" + key;
    }

    edm::LogInfo("L1-O2O") << std::endl << log << std::endl;
  }
}

// ------------ method called once each job just before starting event loop  ------------
void L1O2OTestAnalyzer::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void L1O2OTestAnalyzer::endJob() {}

//define this as a plug-in
DEFINE_FWK_MODULE(L1O2OTestAnalyzer);
