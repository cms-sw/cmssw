#include "L1Trigger/GlobalCaloTrigger/test/gctTestUsingLhcData.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/Common/interface/Handle.h"

#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GlobalCaloTrigger.h"

#include <iostream>

gctTestUsingLhcData::gctTestUsingLhcData() {}
gctTestUsingLhcData::~gctTestUsingLhcData() {}

// Read the region Et values for a single event from a text file and prepare them to be loaded into the GCT
std::vector<L1CaloRegion> gctTestUsingLhcData::loadEvent(const edm::Event& iEvent, const int16_t bx) {
  std::vector<L1CaloRegion> result;

  edm::InputTag inputDataTag("l1GctHwDigis");

  edm::Handle<std::vector<L1CaloRegion> > inputRegions;
  iEvent.getByLabel(inputDataTag, inputRegions);

  for (std::vector<L1CaloRegion>::const_iterator reg = inputRegions->begin(); reg != inputRegions->end(); reg++) {
    if (reg->bx() == bx)
      result.push_back(*reg);
  }
  return result;
}

void gctTestUsingLhcData::checkHwResults(const L1GlobalCaloTrigger* gct, const edm::Event& iEvent) {
  edm::InputTag hwTag("l1GctHwDigis");
  bool passed = checkResults(gct, iEvent, hwTag);
  std::cout << "Check against hardware results " << (passed ? "ok" : "FAILED") << std::endl;
}

void gctTestUsingLhcData::checkEmResults(const L1GlobalCaloTrigger* gct, const edm::Event& iEvent) {
  edm::InputTag emTag("valGctDigis");
  bool passed = checkResults(gct, iEvent, emTag);
  std::cout << "Check against emulator results " << (passed ? "ok" : "FAILED") << std::endl;
}

bool gctTestUsingLhcData::checkResults(const L1GlobalCaloTrigger* gct,
                                       const edm::Event& iEvent,
                                       const edm::InputTag tag) {
  bool checkPassed = true;
  checkPassed &= checkJets(gct, iEvent, tag);
  checkPassed &= checkEtSums(gct, iEvent, tag);
  checkPassed &= checkHtSums(gct, iEvent, tag);
  return checkPassed;
}

bool gctTestUsingLhcData::checkJets(const L1GlobalCaloTrigger* gct, const edm::Event& iEvent, const edm::InputTag tag) {
  edm::InputTag cenJetsTag(tag.label(), "cenJets");
  edm::Handle<L1GctJetCandCollection> cenJetsColl;
  iEvent.getByLabel(cenJetsTag, cenJetsColl);

  edm::InputTag tauJetsTag(tag.label(), "tauJets");
  edm::Handle<L1GctJetCandCollection> tauJetsColl;
  iEvent.getByLabel(tauJetsTag, tauJetsColl);

  edm::InputTag forJetsTag(tag.label(), "forJets");
  edm::Handle<L1GctJetCandCollection> forJetsColl;
  iEvent.getByLabel(forJetsTag, forJetsColl);

  bool match = true;

  match &= (cenJetsColl->size() == gct->getCentralJets().size());
  match &= (tauJetsColl->size() == gct->getTauJets().size());
  match &= (forJetsColl->size() == gct->getForwardJets().size());

  if (match) {
    L1GctJetCandCollection::const_iterator j1, j2;
    L1GctJetCandCollection jetsFromThisGct;

    jetsFromThisGct = gct->getCentralJets();
    for (j1 = cenJetsColl->begin(), j2 = jetsFromThisGct.begin();
         j1 != cenJetsColl->end() && j2 != jetsFromThisGct.end();
         j1++, j2++) {
      if ((*j1) != (*j2)) {
        std::cout << "Jet mismatch; read from file: " << *j1 << "\nFrom this gct: " << *j2 << std::endl;
        match = false;
      }
    }

    jetsFromThisGct = gct->getTauJets();
    for (j1 = tauJetsColl->begin(), j2 = jetsFromThisGct.begin();
         j1 != tauJetsColl->end() && j2 != jetsFromThisGct.end();
         j1++, j2++) {
      if ((*j1) != (*j2)) {
        std::cout << "Jet mismatch; read from file: " << *j1 << "\nFrom this gct: " << *j2 << std::endl;
        match = false;
      }
    }

    jetsFromThisGct = gct->getForwardJets();
    for (j1 = forJetsColl->begin(), j2 = jetsFromThisGct.begin();
         j1 != forJetsColl->end() && j2 != jetsFromThisGct.end();
         j1++, j2++) {
      if ((*j1) != (*j2)) {
        std::cout << "Jet mismatch; read from file: " << *j1 << "\nFrom this gct: " << *j2 << std::endl;
        match = false;
      }
    }

  } else {
    std::cout << "Jet array size check failed!" << std::endl;
  }
  if (!match)
    std::cout << "Jet match checks FAILED" << std::endl;
  return match;
}

bool gctTestUsingLhcData::checkEtSums(const L1GlobalCaloTrigger* gct,
                                      const edm::Event& iEvent,
                                      const edm::InputTag tag) {
  edm::Handle<L1GctEtTotalCollection> ETTColl;
  iEvent.getByLabel(tag, ETTColl);

  edm::Handle<L1GctEtMissCollection> ETMColl;
  iEvent.getByLabel(tag, ETMColl);

  L1GctEtTotalCollection ETTFromThisGct = gct->getEtSumCollection();
  L1GctEtMissCollection ETMFromThisGct = gct->getEtMissCollection();

  L1GctEtTotalCollection::const_iterator ett1, ett2;
  L1GctEtMissCollection::const_iterator etm1, etm2;

  bool match = true;
  for (ett1 = ETTColl->begin(), ett2 = ETTFromThisGct.begin(); ett1 != ETTColl->end() && ett2 != ETTFromThisGct.end();
       ett1++, ett2++) {
    match &= ((*ett1) == (*ett2));
  }
  for (etm1 = ETMColl->begin(), etm2 = ETMFromThisGct.begin(); etm1 != ETMColl->end() && etm2 != ETMFromThisGct.end();
       etm1++, etm2++) {
    match &= ((*etm1) == (*etm2));
  }
  if (!match)
    std::cout << "Et sum match checks FAILED" << std::endl;
  return match;
}

bool gctTestUsingLhcData::checkHtSums(const L1GlobalCaloTrigger* gct,
                                      const edm::Event& iEvent,
                                      const edm::InputTag tag) {
  edm::Handle<L1GctEtHadCollection> HTTColl;
  iEvent.getByLabel(tag, HTTColl);

  edm::Handle<L1GctHtMissCollection> HTMColl;
  iEvent.getByLabel(tag, HTMColl);

  L1GctEtHadCollection HTTFromThisGct = gct->getEtHadCollection();
  L1GctHtMissCollection HTMFromThisGct = gct->getHtMissCollection();

  L1GctEtHadCollection::const_iterator htt1, htt2;
  L1GctHtMissCollection::const_iterator htm1, htm2;

  bool match = true;
  for (htt1 = HTTColl->begin(), htt2 = HTTFromThisGct.begin(); htt1 != HTTColl->end() && htt2 != HTTFromThisGct.end();
       htt1++, htt2++) {
    match &= ((*htt1) == (*htt2));
    if ((*htt1) != (*htt2)) {
      std::cout << "HTT from file " << *htt1 << "\nHTT from gct  " << *htt2 << std::endl;
    }
  }
  for (htm1 = HTMColl->begin(), htm2 = HTMFromThisGct.begin(); htm1 != HTMColl->end() && htm2 != HTMFromThisGct.end();
       htm1++, htm2++) {
    match &= ((*htm1) == (*htm2));
    if ((*htm1) != (*htm2)) {
      std::cout << "HTM from file " << *htm1 << "\nHTM from gct  " << *htm2 << std::endl;
    }
  }
  if (!match)
    std::cout << "Ht sum match checks FAILED" << std::endl;
  return match;
}
