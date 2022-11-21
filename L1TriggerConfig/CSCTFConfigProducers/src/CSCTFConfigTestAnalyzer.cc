// -*- C++ -*-
//
// Package:    CSCTFConfigTestAnalyzer
// Class:      CSCTFConfigTestAnalyzer
//

// system include files
#include <iostream>
#include <memory>

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

//
// class decleration
//

class CSCTFConfigTestAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit CSCTFConfigTestAnalyzer(const edm::ParameterSet&);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // ----------member data ---------------------------
  edm::ESGetToken<L1TriggerKeyList, L1TriggerKeyListRcd> keyListToken_;
  edm::ESGetToken<L1TriggerKey, L1TriggerKeyRcd> keyToken_;
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
CSCTFConfigTestAnalyzer::CSCTFConfigTestAnalyzer(const edm::ParameterSet& iConfig) : keyToken_(esConsumes()) {
  //now do what ever initialization is needed
}

//
// member functions
//

// ------------ method called to for each event  ------------
void CSCTFConfigTestAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  ESHandle<L1TriggerKeyList> pList = iSetup.getHandle(keyListToken_);

  std::cout << "Found " << pList->tscKeyToTokenMap().size() << " TSC keys:" << std::endl;

  L1TriggerKeyList::KeyToToken::const_iterator iTSCKey = pList->tscKeyToTokenMap().begin();
  L1TriggerKeyList::KeyToToken::const_iterator eTSCKey = pList->tscKeyToTokenMap().end();
  for (; iTSCKey != eTSCKey; ++iTSCKey) {
    std::cout << iTSCKey->first << " " << iTSCKey->second << std::endl;
  }
  std::cout << std::endl;

  L1TriggerKeyList::RecordToKeyToToken::const_iterator iRec = pList->recordTypeToKeyToTokenMap().begin();
  L1TriggerKeyList::RecordToKeyToToken::const_iterator eRec = pList->recordTypeToKeyToTokenMap().end();
  for (; iRec != eRec; ++iRec) {
    const L1TriggerKeyList::KeyToToken& keyTokenMap = iRec->second;
    std::cout << "For record@type " << iRec->first << ", found " << keyTokenMap.size() << " keys:" << std::endl;

    L1TriggerKeyList::KeyToToken::const_iterator iKey = keyTokenMap.begin();
    L1TriggerKeyList::KeyToToken::const_iterator eKey = keyTokenMap.end();
    for (; iKey != eKey; ++iKey) {
      std::cout << iKey->first << " " << iKey->second << std::endl;
    }
    std::cout << std::endl;
  }

  try {
    ESHandle<L1TriggerKey> pKey = iSetup.getHandle(keyToken_);

    // std::cout << "Current TSC key = " << pKey->getTSCKey() << std::endl ;
    std::cout << "Current TSC key = " << pKey->tscKey() << std::endl;

    std::cout << "Current subsystem keys:" << std::endl;
    std::cout << "CSCTF " << pKey->subsystemKey(L1TriggerKey::kCSCTF) << std::endl;
    std::cout << "DTTF " << pKey->subsystemKey(L1TriggerKey::kDTTF) << std::endl;
    std::cout << "RPC " << pKey->subsystemKey(L1TriggerKey::kRPC) << std::endl;
    std::cout << "GMT " << pKey->subsystemKey(L1TriggerKey::kGMT) << std::endl;
    std::cout << "RCT " << pKey->subsystemKey(L1TriggerKey::kRCT) << std::endl;
    std::cout << "GCT " << pKey->subsystemKey(L1TriggerKey::kGCT) << std::endl;
    std::cout << "TSP0 " << pKey->subsystemKey(L1TriggerKey::kTSP0) << std::endl;

    const L1TriggerKey::RecordToKey& recKeyMap = pKey->recordToKeyMap();
    L1TriggerKey::RecordToKey::const_iterator iRec = recKeyMap.begin();
    L1TriggerKey::RecordToKey::const_iterator eRec = recKeyMap.end();
    for (; iRec != eRec; ++iRec) {
      std::cout << iRec->first << " " << iRec->second << std::endl;
    }
  } catch (cms::Exception& ex) {
    std::cout << "No L1TriggerKey found." << std::endl;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(CSCTFConfigTestAnalyzer);
