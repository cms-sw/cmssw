#include <string>
#include <list>
#include <vector>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <cassert>
#include <utility>

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/L1TGlobal/interface/GlobalObjectMapFwd.h"
#include "DataFormats/L1TGlobal/interface/GlobalObjectMap.h"
#include "DataFormats/L1TGlobal/interface/GlobalObjectMapRecord.h"
#include "DataFormats/L1TGlobal/interface/GlobalObject.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"

#include "HLTL1TSeed.h"

using namespace std;

// constructors
HLTL1TSeed::HLTL1TSeed(const edm::ParameterSet& parSet)
    : HLTStreamFilter(parSet),
      //useObjectMaps_(parSet.getParameter<bool>("L1UseL1TriggerObjectMaps")),
      m_l1SeedsLogicalExpression(parSet.getParameter<string>("L1SeedsLogicalExpression")),
      m_l1GtObjectMapTag(parSet.getParameter<edm::InputTag>("L1ObjectMapInputTag")),
      m_l1GtObjectMapToken(consumes<GlobalObjectMapRecord>(m_l1GtObjectMapTag)),
      m_l1GlobalTag(parSet.getParameter<edm::InputTag>("L1GlobalInputTag")),
      m_l1GlobalToken(consumes<GlobalAlgBlkBxCollection>(m_l1GlobalTag)),
      m_l1MuonCollectionsTag(parSet.getParameter<edm::InputTag>("L1MuonInputTag")),  // FIX WHEN UNPACKERS ADDED
      m_l1MuonTag(m_l1MuonCollectionsTag),
      m_l1MuonToken(consumes<l1t::MuonBxCollection>(m_l1MuonTag)),
      m_l1MuonShowerCollectionsTag(
          parSet.getParameter<edm::InputTag>("L1MuonShowerInputTag")),  // FIX WHEN UNPACKERS ADDED
      m_l1MuonShowerTag(m_l1MuonShowerCollectionsTag),
      m_l1MuonShowerToken(consumes<l1t::MuonShowerBxCollection>(m_l1MuonShowerTag)),
      m_l1EGammaCollectionsTag(parSet.getParameter<edm::InputTag>("L1EGammaInputTag")),  // FIX WHEN UNPACKERS ADDED
      m_l1EGammaTag(m_l1EGammaCollectionsTag),
      m_l1EGammaToken(consumes<l1t::EGammaBxCollection>(m_l1EGammaTag)),
      m_l1JetCollectionsTag(parSet.getParameter<edm::InputTag>("L1JetInputTag")),  // FIX WHEN UNPACKERS ADDED
      m_l1JetTag(m_l1JetCollectionsTag),
      m_l1JetToken(consumes<l1t::JetBxCollection>(m_l1JetTag)),
      m_l1TauCollectionsTag(parSet.getParameter<edm::InputTag>("L1TauInputTag")),  // FIX WHEN UNPACKERS ADDED
      m_l1TauTag(m_l1TauCollectionsTag),
      m_l1TauToken(consumes<l1t::TauBxCollection>(m_l1TauTag)),
      m_l1EtSumCollectionsTag(parSet.getParameter<edm::InputTag>("L1EtSumInputTag")),  // FIX WHEN UNPACKERS ADDED
      m_l1EtSumTag(m_l1EtSumCollectionsTag),
      m_l1EtSumToken(consumes<l1t::EtSumBxCollection>(m_l1EtSumTag)),
      m_l1EtSumZdcCollectionsTag(parSet.getParameter<edm::InputTag>("L1EtSumZdcInputTag")),  // FIX WHEN UNPACKERS ADDED
      m_l1EtSumZdcTag(m_l1EtSumZdcCollectionsTag),
      m_l1EtSumZdcToken(consumes<l1t::EtSumBxCollection>(m_l1EtSumZdcTag)),
      m_l1GlobalDecision(false),
      m_isDebugEnabled(edm::isDebugEnabled()) {
  if (m_l1SeedsLogicalExpression.empty()) {
    throw cms::Exception("FailModule") << "\nTrying to seed with an empty L1SeedsLogicalExpression.\n" << std::endl;

  } else if (m_l1SeedsLogicalExpression != "L1GlobalDecision") {
    // check also the logical expression - add/remove spaces if needed
    m_l1AlgoLogicParser = GlobalLogicParser(m_l1SeedsLogicalExpression);

    // list of required algorithms for seeding
    // dummy values for tokenNumber and tokenResult
    m_l1AlgoSeeds.reserve((m_l1AlgoLogicParser.operandTokenVector()).size());
    m_l1AlgoSeeds = m_l1AlgoLogicParser.expressionSeedsOperandList();
    size_t const l1AlgoSeedsSize = m_l1AlgoSeeds.size();

    m_l1AlgoSeedsRpn.reserve(l1AlgoSeedsSize);
    m_l1AlgoSeedsObjType.reserve(l1AlgoSeedsSize);

  } else {
    m_l1GlobalDecision = true;
  }
}

void HLTL1TSeed::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);

  // # logical expression for the required L1 algorithms;
  // # the algorithms are specified by name
  // # allowed operators: "AND", "OR", "NOT", "(", ")"
  // #
  // # by convention, "L1GlobalDecision" logical expression means global decision
  desc.add<string>("L1SeedsLogicalExpression", "");
  desc.add<edm::InputTag>("L1ObjectMapInputTag", edm::InputTag("hltGtStage2ObjectMap"));
  desc.add<edm::InputTag>("L1GlobalInputTag", edm::InputTag("hltGtStage2Digis"));
  desc.add<edm::InputTag>("L1MuonInputTag", edm::InputTag("hltGtStage2Digis:Muon"));
  desc.add<edm::InputTag>("L1MuonShowerInputTag", edm::InputTag("hltGtStage2Digis:MuonShower"));
  desc.add<edm::InputTag>("L1EGammaInputTag", edm::InputTag("hltGtStage2Digis:EGamma"));
  desc.add<edm::InputTag>("L1JetInputTag", edm::InputTag("hltGtStage2Digis:Jet"));
  desc.add<edm::InputTag>("L1TauInputTag", edm::InputTag("hltGtStage2Digis:Tau"));
  desc.add<edm::InputTag>("L1EtSumInputTag", edm::InputTag("hltGtStage2Digis:EtSum"));
  desc.add<edm::InputTag>("L1EtSumZdcInputTag", edm::InputTag("hltGtStage2Digis:EtSumZDC"));
  descriptions.add("hltL1TSeed", desc);
}

bool HLTL1TSeed::hltFilter(edm::Event& iEvent,
                           const edm::EventSetup& evSetup,
                           trigger::TriggerFilterObjectWithRefs& filterproduct) {
  bool rc = false;

  // the filter object
  if (saveTags()) {
    // muons
    filterproduct.addCollectionTag(m_l1MuonTag);

    // muon showers
    filterproduct.addCollectionTag(m_l1MuonShowerTag);

    // egamma
    filterproduct.addCollectionTag(m_l1EGammaTag);

    // jet
    filterproduct.addCollectionTag(m_l1JetTag);

    // tau
    filterproduct.addCollectionTag(m_l1TauTag);

    // etsum
    filterproduct.addCollectionTag(m_l1EtSumTag);

    // etsum (ZDC)
    filterproduct.addCollectionTag(m_l1EtSumZdcTag);
  }

  // Get all the seeding from iEvent (i.e. L1TriggerObjectMapRecord)
  //
  rc = seedsL1TriggerObjectMaps(iEvent, filterproduct);

  if (m_isDebugEnabled) {
    dumpTriggerFilterObjectWithRefs(filterproduct);
  }

  return rc;
}

// detailed print of filter content
void HLTL1TSeed::dumpTriggerFilterObjectWithRefs(trigger::TriggerFilterObjectWithRefs& filterproduct) const {
  LogTrace("HLTL1TSeed") << "\nHLTL1TSeed::hltFilter "
                         << "\n  Dump TriggerFilterObjectWithRefs\n"
                         << endl;

  vector<l1t::MuonRef> seedsL1Mu;
  filterproduct.getObjects(trigger::TriggerL1Mu, seedsL1Mu);
  const size_t sizeSeedsL1Mu = seedsL1Mu.size();

  LogTrace("HLTL1TSeed") << "\n  HLTL1TSeed: seed logical expression = " << m_l1SeedsLogicalExpression << endl;

  LogTrace("HLTL1TSeed") << "\n  L1Mu seeds:      " << sizeSeedsL1Mu << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1Mu; i++) {
    l1t::MuonRef obj = l1t::MuonRef(seedsL1Mu[i]);

    LogTrace("HLTL1TSeed") << "\tL1Mu     "
                           << "\t"
                           << "q = "
                           << obj->hwCharge()  // TEMP get hwCharge insead of charge which is not yet set NEED FIX.
                           << "\t"
                           << "pt = " << obj->pt() << "\t"
                           << "eta =  " << obj->eta() << "\t"
                           << "phi =  " << obj->phi();  //<< "\t" << "BX = " << obj->bx();
  }

  vector<l1t::MuonShowerRef> seedsL1MuShower;
  filterproduct.getObjects(trigger::TriggerL1MuShower, seedsL1MuShower);
  const size_t sizeSeedsL1MuShower = seedsL1MuShower.size();

  LogTrace("HLTL1TSeed") << "\n  HLTL1TSeed: seed logical expression = " << m_l1SeedsLogicalExpression << endl;

  LogTrace("HLTL1TSeed") << "\n  L1MuShower seeds:      " << sizeSeedsL1MuShower << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1MuShower; i++) {
    l1t::MuonShowerRef obj = l1t::MuonShowerRef(seedsL1MuShower[i]);

    LogTrace("HLTL1TSeed") << "\tL1MuShower     "
                           << "\t"
                           << "pt = " << obj->pt() << "\t"
                           << "eta =  " << obj->eta() << "\t"
                           << "phi =  " << obj->phi();  //<< "\t" << "BX = " << obj->bx();
  }

  vector<l1t::EGammaRef> seedsL1EG;
  filterproduct.getObjects(trigger::TriggerL1EG, seedsL1EG);
  const size_t sizeSeedsL1EG = seedsL1EG.size();

  LogTrace("HLTL1TSeed") << "\n  L1EG seeds:      " << sizeSeedsL1EG << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1EG; i++) {
    l1t::EGammaRef obj = l1t::EGammaRef(seedsL1EG[i]);

    LogTrace("HLTL1TSeed") << "\tL1EG     "
                           << "\t"
                           << "pt = " << obj->pt() << "\t"
                           << "eta =  " << obj->eta() << "\t"
                           << "phi =  " << obj->phi();  //<< "\t" << "BX = " << obj->bx();
  }

  vector<l1t::JetRef> seedsL1Jet;
  filterproduct.getObjects(trigger::TriggerL1Jet, seedsL1Jet);
  const size_t sizeSeedsL1Jet = seedsL1Jet.size();

  LogTrace("HLTL1TSeed") << "\n  L1Jet seeds:      " << sizeSeedsL1Jet << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1Jet; i++) {
    l1t::JetRef obj = l1t::JetRef(seedsL1Jet[i]);

    LogTrace("HLTL1TSeed") << "\tL1Jet     "
                           << "\t"
                           << "pt = " << obj->pt() << "\t"
                           << "eta =  " << obj->eta() << "\t"
                           << "phi =  " << obj->phi();  //<< "\t" << "BX = " << obj->bx();
  }

  vector<l1t::TauRef> seedsL1Tau;
  filterproduct.getObjects(trigger::TriggerL1Tau, seedsL1Tau);
  const size_t sizeSeedsL1Tau = seedsL1Tau.size();

  LogTrace("HLTL1TSeed") << "\n  L1Tau seeds:      " << sizeSeedsL1Tau << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1Tau; i++) {
    l1t::TauRef obj = l1t::TauRef(seedsL1Tau[i]);

    LogTrace("HLTL1TSeed") << "\tL1Tau     "
                           << "\t"
                           << "pt = " << obj->pt() << "\t"
                           << "eta =  " << obj->eta() << "\t"
                           << "phi =  " << obj->phi();  //<< "\t" << "BX = " << obj->bx();
  }

  vector<l1t::EtSumRef> seedsL1EtSumETT;
  filterproduct.getObjects(trigger::TriggerL1ETT, seedsL1EtSumETT);
  const size_t sizeSeedsL1EtSumETT = seedsL1EtSumETT.size();
  LogTrace("HLTL1TSeed") << "\n  L1EtSum ETT seeds:      " << sizeSeedsL1EtSumETT << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1EtSumETT; i++) {
    l1t::EtSumRef obj = l1t::EtSumRef(seedsL1EtSumETT[i]);

    LogTrace("HLTL1TSeed") << "\tL1EtSum  ETT"
                           << "\t"
                           << "pt = " << obj->pt() << "\t"
                           << "eta =  " << obj->eta() << "\t"
                           << "phi =  " << obj->phi();  //<< "\t" << "BX = " << obj->bx();
  }

  vector<l1t::EtSumRef> seedsL1EtSumHTT;
  filterproduct.getObjects(trigger::TriggerL1HTT, seedsL1EtSumHTT);
  const size_t sizeSeedsL1EtSumHTT = seedsL1EtSumHTT.size();
  LogTrace("HLTL1TSeed") << "\n  L1EtSum HTT seeds:      " << sizeSeedsL1EtSumHTT << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1EtSumHTT; i++) {
    l1t::EtSumRef obj = l1t::EtSumRef(seedsL1EtSumHTT[i]);

    LogTrace("HLTL1TSeed") << "\tL1EtSum  HTT"
                           << "\t"
                           << "pt = " << obj->pt() << "\t"
                           << "eta =  " << obj->eta() << "\t"
                           << "phi =  " << obj->phi();  //<< "\t" << "BX = " << obj->bx();
  }

  vector<l1t::EtSumRef> seedsL1EtSumETM;
  filterproduct.getObjects(trigger::TriggerL1ETM, seedsL1EtSumETM);
  const size_t sizeSeedsL1EtSumETM = seedsL1EtSumETM.size();
  LogTrace("HLTL1TSeed") << "\n  L1EtSum ETM seeds:      " << sizeSeedsL1EtSumETM << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1EtSumETM; i++) {
    l1t::EtSumRef obj = l1t::EtSumRef(seedsL1EtSumETM[i]);

    LogTrace("HLTL1TSeed") << "\tL1EtSum  ETM"
                           << "\t"
                           << "pt = " << obj->pt() << "\t"
                           << "eta =  " << obj->eta() << "\t"
                           << "phi =  " << obj->phi();  //<< "\t" << "BX = " << obj->bx();
  }

  vector<l1t::EtSumRef> seedsL1EtSumETMHF;
  filterproduct.getObjects(trigger::TriggerL1ETMHF, seedsL1EtSumETMHF);
  const size_t sizeSeedsL1EtSumETMHF = seedsL1EtSumETMHF.size();
  LogTrace("HLTL1TSeed") << "\n  L1EtSum ETMHF seeds:      " << sizeSeedsL1EtSumETMHF << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1EtSumETMHF; i++) {
    l1t::EtSumRef obj = l1t::EtSumRef(seedsL1EtSumETMHF[i]);

    LogTrace("HLTL1TSeed") << "\tL1EtSum  ETMHF"
                           << "\t"
                           << "pt = " << obj->pt() << "\t"
                           << "eta =  " << obj->eta() << "\t"
                           << "phi =  " << obj->phi();  //<< "\t" << "BX = " << obj->bx();
  }

  vector<l1t::EtSumRef> seedsL1EtSumHTMHF;
  filterproduct.getObjects(trigger::TriggerL1HTMHF, seedsL1EtSumHTMHF);
  const size_t sizeSeedsL1EtSumHTMHF = seedsL1EtSumHTMHF.size();
  LogTrace("HLTL1TSeed") << "\n  L1EtSum HTMHF seeds:      " << sizeSeedsL1EtSumHTMHF << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1EtSumHTMHF; i++) {
    l1t::EtSumRef obj = l1t::EtSumRef(seedsL1EtSumHTMHF[i]);

    LogTrace("HLTL1TSeed") << "\tL1EtSum  HTMHF"
                           << "\t"
                           << "pt = " << obj->pt() << "\t"
                           << "eta =  " << obj->eta() << "\t"
                           << "phi =  " << obj->phi();  //<< "\t" << "BX = " << obj->bx();
  }

  vector<l1t::EtSumRef> seedsL1EtSumHTM;
  filterproduct.getObjects(trigger::TriggerL1HTM, seedsL1EtSumHTM);
  const size_t sizeSeedsL1EtSumHTM = seedsL1EtSumHTM.size();
  LogTrace("HLTL1TSeed") << "\n  L1EtSum HTM seeds:      " << sizeSeedsL1EtSumHTM << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1EtSumHTM; i++) {
    l1t::EtSumRef obj = l1t::EtSumRef(seedsL1EtSumHTM[i]);

    LogTrace("HLTL1TSeed") << "\tL1EtSum  HTM"
                           << "\t"
                           << "pt = " << obj->pt() << "\t"
                           << "eta =  " << obj->eta() << "\t"
                           << "phi =  " << obj->phi();  //<< "\t" << "BX = " << obj->bx();
  }

  vector<l1t::EtSumRef> seedsL1EtSumCentrality;
  filterproduct.getObjects(trigger::TriggerL1Centrality, seedsL1EtSumCentrality);
  const size_t sizeSeedsL1EtSumCentrality = seedsL1EtSumCentrality.size();
  LogTrace("HLTL1TSeed") << "\n  L1EtSum Centrality seeds:      " << sizeSeedsL1EtSumCentrality << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1EtSumCentrality; i++) {
    l1t::EtSumRef obj = l1t::EtSumRef(seedsL1EtSumCentrality[i]);

    LogTrace("HLTL1TSeed") << "\tL1EtSum  Centrality Bits: " << std::bitset<8>(obj->hwPt())
                           << " (hwPt = " << obj->hwPt() << ")";
  }

  vector<l1t::EtSumRef> seedsL1EtSumMinBiasHFP0;
  filterproduct.getObjects(trigger::TriggerL1MinBiasHFP0, seedsL1EtSumMinBiasHFP0);
  const size_t sizeSeedsL1EtSumMinBiasHFP0 = seedsL1EtSumMinBiasHFP0.size();
  LogTrace("HLTL1TSeed") << "\n  L1EtSum MinBiasHFP0 seeds:      " << sizeSeedsL1EtSumMinBiasHFP0 << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1EtSumMinBiasHFP0; i++) {
    l1t::EtSumRef obj = l1t::EtSumRef(seedsL1EtSumMinBiasHFP0[i]);

    LogTrace("HLTL1TSeed") << "\tL1EtSum  MinBiasHFP0: hwPt = " << obj->hwPt();
  }

  vector<l1t::EtSumRef> seedsL1EtSumMinBiasHFM0;
  filterproduct.getObjects(trigger::TriggerL1MinBiasHFM0, seedsL1EtSumMinBiasHFM0);
  const size_t sizeSeedsL1EtSumMinBiasHFM0 = seedsL1EtSumMinBiasHFM0.size();
  LogTrace("HLTL1TSeed") << "\n  L1EtSum MinBiasHFM0 seeds:      " << sizeSeedsL1EtSumMinBiasHFM0 << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1EtSumMinBiasHFM0; i++) {
    l1t::EtSumRef obj = l1t::EtSumRef(seedsL1EtSumMinBiasHFM0[i]);

    LogTrace("HLTL1TSeed") << "\tL1EtSum  MinBiasHFM0: hwPt = " << obj->hwPt();
  }

  vector<l1t::EtSumRef> seedsL1EtSumMinBiasHFP1;
  filterproduct.getObjects(trigger::TriggerL1MinBiasHFP1, seedsL1EtSumMinBiasHFP1);
  const size_t sizeSeedsL1EtSumMinBiasHFP1 = seedsL1EtSumMinBiasHFP1.size();
  LogTrace("HLTL1TSeed") << "\n  L1EtSum MinBiasHFP1 seeds:      " << sizeSeedsL1EtSumMinBiasHFP1 << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1EtSumMinBiasHFP1; i++) {
    l1t::EtSumRef obj = l1t::EtSumRef(seedsL1EtSumMinBiasHFP1[i]);

    LogTrace("HLTL1TSeed") << "\tL1EtSum  MinBiasHFP1: hwPt = " << obj->hwPt();
  }

  vector<l1t::EtSumRef> seedsL1EtSumMinBiasHFM1;
  filterproduct.getObjects(trigger::TriggerL1MinBiasHFM1, seedsL1EtSumMinBiasHFM1);
  const size_t sizeSeedsL1EtSumMinBiasHFM1 = seedsL1EtSumMinBiasHFM1.size();
  LogTrace("HLTL1TSeed") << "\n  L1EtSum MinBiasHFM1 seeds:      " << sizeSeedsL1EtSumMinBiasHFM1 << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1EtSumMinBiasHFM1; i++) {
    l1t::EtSumRef obj = l1t::EtSumRef(seedsL1EtSumMinBiasHFM1[i]);

    LogTrace("HLTL1TSeed") << "\tL1EtSum  MinBiasHFM1: hwPt = " << obj->hwPt();
  }

  vector<l1t::EtSumRef> seedsL1EtSumTowerCount;
  filterproduct.getObjects(trigger::TriggerL1TowerCount, seedsL1EtSumTowerCount);
  const size_t sizeSeedsL1EtSumTowerCount = seedsL1EtSumTowerCount.size();
  LogTrace("HLTL1TSeed") << "\n  L1EtSum TowerCount seeds:      " << sizeSeedsL1EtSumTowerCount << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1EtSumTowerCount; i++) {
    l1t::EtSumRef obj = l1t::EtSumRef(seedsL1EtSumTowerCount[i]);

    LogTrace("HLTL1TSeed") << "\tL1EtSum  TowerCount: hwPt = " << obj->hwPt();
  }

  vector<l1t::EtSumRef> seedsL1EtSumAsymEt;
  filterproduct.getObjects(trigger::TriggerL1AsymEt, seedsL1EtSumAsymEt);
  const size_t sizeSeedsL1EtSumAsymEt = seedsL1EtSumAsymEt.size();
  LogTrace("HLTL1TSeed") << "\n  L1EtSum AsymEt seeds:      " << sizeSeedsL1EtSumAsymEt << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1EtSumAsymEt; i++) {
    l1t::EtSumRef obj = l1t::EtSumRef(seedsL1EtSumAsymEt[i]);

    LogTrace("HLTL1TSeed") << "\tL1EtSum  AsymEt: hwPt = " << obj->hwPt();
  }

  vector<l1t::EtSumRef> seedsL1EtSumAsymHt;
  filterproduct.getObjects(trigger::TriggerL1AsymHt, seedsL1EtSumAsymHt);
  const size_t sizeSeedsL1EtSumAsymHt = seedsL1EtSumAsymHt.size();
  LogTrace("HLTL1TSeed") << "\n  L1EtSum AsymHt seeds:      " << sizeSeedsL1EtSumAsymHt << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1EtSumAsymHt; i++) {
    l1t::EtSumRef obj = l1t::EtSumRef(seedsL1EtSumAsymHt[i]);

    LogTrace("HLTL1TSeed") << "\tL1EtSum  AsymHt: hwPt = " << obj->hwPt();
  }

  vector<l1t::EtSumRef> seedsL1EtSumAsymEtHF;
  filterproduct.getObjects(trigger::TriggerL1AsymEtHF, seedsL1EtSumAsymEtHF);
  const size_t sizeSeedsL1EtSumAsymEtHF = seedsL1EtSumAsymEtHF.size();
  LogTrace("HLTL1TSeed") << "\n  L1EtSum AsymEtHF seeds:      " << sizeSeedsL1EtSumAsymEtHF << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1EtSumAsymEtHF; i++) {
    l1t::EtSumRef obj = l1t::EtSumRef(seedsL1EtSumAsymEtHF[i]);

    LogTrace("HLTL1TSeed") << "\tL1EtSum  AsymEtHF: hwPt = " << obj->hwPt();
  }

  vector<l1t::EtSumRef> seedsL1EtSumAsymHtHF;
  filterproduct.getObjects(trigger::TriggerL1AsymHtHF, seedsL1EtSumAsymHtHF);
  const size_t sizeSeedsL1EtSumAsymHtHF = seedsL1EtSumAsymHtHF.size();
  LogTrace("HLTL1TSeed") << "\n  L1EtSum AsymHtHF seeds:      " << sizeSeedsL1EtSumAsymHtHF << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1EtSumAsymHtHF; i++) {
    l1t::EtSumRef obj = l1t::EtSumRef(seedsL1EtSumAsymHtHF[i]);

    LogTrace("HLTL1TSeed") << "\tL1EtSum  AsymHtHF: hwPt = " << obj->hwPt();
  }

  vector<l1t::EtSumRef> seedsL1EtSumZDCP;
  filterproduct.getObjects(trigger::TriggerL1ZDCP, seedsL1EtSumZDCP);
  const size_t sizeSeedsL1EtSumZDCP = seedsL1EtSumZDCP.size();
  LogTrace("HLTL1TSeed") << "\n  L1EtSum ZDCP seeds:      " << sizeSeedsL1EtSumZDCP << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1EtSumZDCP; i++) {
    l1t::EtSumRef obj = l1t::EtSumRef(seedsL1EtSumZDCP[i]);

    LogTrace("HLTL1TSeed") << "\tL1EtSum  ZDCP"
                           << "\t"
                           << "pt = " << obj->pt() << "\t"
                           << "eta =  " << obj->eta() << "\t"
                           << "phi =  " << obj->phi();  //<< "\t" << "BX = " << obj->bx();
  }

  vector<l1t::EtSumRef> seedsL1EtSumZDCM;
  filterproduct.getObjects(trigger::TriggerL1ZDCM, seedsL1EtSumZDCM);
  const size_t sizeSeedsL1EtSumZDCM = seedsL1EtSumZDCM.size();
  LogTrace("HLTL1TSeed") << "\n  L1EtSum ZDCM seeds:      " << sizeSeedsL1EtSumZDCM << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1EtSumZDCM; i++) {
    l1t::EtSumRef obj = l1t::EtSumRef(seedsL1EtSumZDCM[i]);

    LogTrace("HLTL1TSeed") << "\tL1EtSum  ZDCM"
                           << "\t"
                           << "pt = " << obj->pt() << "\t"
                           << "eta =  " << obj->eta() << "\t"
                           << "phi =  " << obj->phi();  //<< "\t" << "BX = " << obj->bx();
  }

  LogTrace("HLTL1TSeed") << " \n\n" << endl;
}

// seeding is done via L1 trigger object maps, considering the objects which fired in L1
bool HLTL1TSeed::seedsL1TriggerObjectMaps(edm::Event& iEvent, trigger::TriggerFilterObjectWithRefs& filterproduct) {
  // Two GT objects are obtained from the Event: (1) the unpacked GT and (2) the emulated GT.
  // Return value of the function is the score of seeding logical expression, evaluated using (1).
  // Seeding is performed (per l1_algo) if ACCEPT both in (1) and (2). Seed objects are identified
  // and only available from ObjectMaps created in (2).

  // define index lists for all particle types
  using idxListType = std::list<std::pair<L1TObjBxIndexType, L1TObjIndexType>>;

  idxListType listMuon;
  idxListType listMuonShower;

  idxListType listEG;

  idxListType listJet;
  idxListType listTau;

  idxListType listETM;
  idxListType listETT;
  idxListType listHTT;
  idxListType listHTM;
  idxListType listETMHF;
  idxListType listHTMHF;

  idxListType listCentrality;
  idxListType listMinBiasHFP0;
  idxListType listMinBiasHFM0;
  idxListType listMinBiasHFP1;
  idxListType listMinBiasHFM1;
  idxListType listTotalEtEm;
  idxListType listMissingEtHF;
  idxListType listTowerCount;
  idxListType listAsymEt;
  idxListType listAsymHt;
  idxListType listAsymEtHF;
  idxListType listAsymHtHF;
  idxListType listZDCP;
  idxListType listZDCM;

  // get handle to unpacked GT
  edm::Handle<GlobalAlgBlkBxCollection> uGtAlgoBlocks;
  iEvent.getByToken(m_l1GlobalToken, uGtAlgoBlocks);

  if (!uGtAlgoBlocks.isValid()) {
    edm::LogWarning("HLTL1TSeed") << " Warning: GlobalAlgBlkBxCollection with input tag " << m_l1GlobalTag
                                  << " requested in configuration, but not found in the event." << std::endl;

    return false;
  }

  // check size (all BXs)
  if (uGtAlgoBlocks->size() == 0) {
    edm::LogWarning("HLTL1TSeed") << " Warning: GlobalAlgBlkBxCollection with input tag " << m_l1GlobalTag
                                  << " is empty for all BXs.";
    return false;
  }

  // check size (BX 0)
  if (uGtAlgoBlocks->isEmpty(0)) {
    edm::LogWarning("HLTL1TSeed") << " Warning: GlobalAlgBlkBxCollection with input tag " << m_l1GlobalTag
                                  << " is empty for BX=0.";
    return false;
  }

  // get handle to object maps from emulator (one object map per algorithm)
  edm::Handle<GlobalObjectMapRecord> gtObjectMapRecord;
  iEvent.getByToken(m_l1GtObjectMapToken, gtObjectMapRecord);

  if (!gtObjectMapRecord.isValid()) {
    edm::LogWarning("HLTL1TSeed") << " Warning: GlobalObjectMapRecord with input tag " << m_l1GtObjectMapTag
                                  << " requested in configuration, but not found in the event." << std::endl;

    return false;
  }

  if (m_isDebugEnabled) {
    const std::vector<GlobalObjectMap>& objMaps = gtObjectMapRecord->gtObjectMap();

    LogTrace("HLTL1TSeed") << "\nHLTL1Seed"
                           << "\n--------------------------------------------------------------------------------------"
                              "-------------------------------";

    LogTrace("HLTL1TSeed")
        << "\n\tAlgorithms in L1TriggerObjectMapRecord and GT results ( emulated | initial | prescaled | final ) "
        << endl;

    LogTrace("HLTL1TSeed") << "\n\tmap"
                           << "\tAlgoBit" << std::setw(40) << "algoName"
                           << "\t (emul|ini|pre|fin)" << endl;

    LogTrace("HLTL1TSeed") << "----------------------------------------------------------------------------------------"
                              "-----------------------------";

    for (size_t imap = 0; imap < objMaps.size(); imap++) {
      int bit = objMaps[imap].algoBitNumber();  //  same as bit from L1T Menu

      int emulDecision = objMaps[imap].algoGtlResult();

      // For bx=0 , get 0th AlgoBlock, so in BXvector at(bx=0,i=0)
      int initDecision = (uGtAlgoBlocks->at(0, 0)).getAlgoDecisionInitial(bit);
      int presDecision = (uGtAlgoBlocks->at(0, 0)).getAlgoDecisionInterm(bit);
      int finlDecision = (uGtAlgoBlocks->at(0, 0)).getAlgoDecisionFinal(bit);

      if (emulDecision != initDecision) {
        LogTrace("HLTL1TSeed") << "L1T decision (emulated vs. unpacked initial) is not the same:"
                               << "\n\tbit = " << std::setw(3) << bit << std::setw(40) << objMaps[imap].algoName()
                               << "\t emulated decision = " << emulDecision
                               << "\t unpacked initial decision = " << initDecision
                               << "\nThis should not happen. Include the L1TGtEmulCompare module in the sequence."
                               << endl;
      }

      LogTrace("HLTL1TSeed") << "\t" << std::setw(3) << imap << "\tbit = " << std::setw(3) << bit << std::setw(40)
                             << objMaps[imap].algoName() << "\t (  " << emulDecision << " | " << initDecision << " | "
                             << presDecision << " | " << finlDecision << " ) ";
    }
    LogTrace("HLTL1TSeed") << endl;
  }

  // Filter decision in case of "L1GlobalDecision" logical expression.
  // By convention, it means global decision.
  // /////////////////////////////////////////////////////////////////
  if (m_l1GlobalDecision) {
    // For bx=0 , get 0th AlgoBlock, so in BXvector at(bx=0,i=0)
    return (uGtAlgoBlocks->at(0, 0)).getFinalOR();
  }

  // Update/Reset m_l1AlgoLogicParser by reseting token result
  // /////////////////////////////////////////////////////////
  std::vector<GlobalLogicParser::OperandToken>& algOpTokenVector = m_l1AlgoLogicParser.operandTokenVector();

  for (auto& i : algOpTokenVector) {
    // rest token result
    //
    i.tokenResult = false;
  }

  // Update m_l1AlgoLogicParser and store emulator results for algOpTokens
  // /////////////////////////////////////////////////////////////////////
  for (auto& i : algOpTokenVector) {
    std::string algoName = i.tokenName;

    const GlobalObjectMap* objMap = gtObjectMapRecord->getObjectMap(algoName);

    if (objMap == nullptr) {
      throw cms::Exception("FailModule")
          << "\nAlgorithm " << algoName
          << ", requested as seed by a HLT path, cannot be matched to a L1 algo name in any GlobalObjectMap\n"
          << "Please check if algorithm " << algoName << " is present in the L1 menu\n"
          << std::endl;

    } else {
      //(algOpTokenVector[i]).tokenResult = objMap->algoGtlResult();

      int bit = objMap->algoBitNumber();
      bool finalAlgoDecision = (uGtAlgoBlocks->at(0, 0)).getAlgoDecisionFinal(bit);
      i.tokenResult = finalAlgoDecision;
    }
  }

  // Filter decision
  // ///////////////
  bool seedsResult = m_l1AlgoLogicParser.expressionResult();

  if (m_isDebugEnabled) {
    LogTrace("HLTL1TSeed") << "\nHLTL1TSeed: l1SeedsLogicalExpression (names) = '" << m_l1SeedsLogicalExpression << "'"
                           << "\n  Result for logical expression after update of algOpTokens: " << seedsResult << "\n"
                           << std::endl;
  }

  /// Loop over the list of required algorithms for seeding
  /// /////////////////////////////////////////////////////

  for (std::vector<GlobalLogicParser::OperandToken>::const_iterator itSeed = m_l1AlgoSeeds.begin();
       itSeed != m_l1AlgoSeeds.end();
       ++itSeed) {
    std::string algoSeedName = (*itSeed).tokenName;

    LogTrace("HLTL1TSeed") << "\n ----------------  algo seed name = " << algoSeedName << endl;

    const GlobalObjectMap* objMap = gtObjectMapRecord->getObjectMap(algoSeedName);

    if (objMap == nullptr) {
      // Should not get here
      //
      throw cms::Exception("FailModule")
          << "\nAlgorithm " << algoSeedName
          << ", requested as seed by a HLT path, cannot be matched to a L1 algo name in any GlobalObjectMap\n"
          << "Please check if algorithm " << algoSeedName << " is present in the L1 menu\n"
          << std::endl;
    }

    int algoSeedBitNumber = objMap->algoBitNumber();
    bool algoSeedResult = objMap->algoGtlResult();

    // unpacked GT results: uGtAlgoBlock has decisions initial, prescaled, and final after masks
    bool algoSeedResultMaskAndPresc = uGtAlgoBlocks->at(0, 0).getAlgoDecisionFinal(algoSeedBitNumber);

    LogTrace("HLTL1TSeed") << "\n\tAlgo seed " << algoSeedName << " result emulated | final = " << algoSeedResult
                           << " | " << algoSeedResultMaskAndPresc << endl;

    /// Unpacked GT result of algorithm is false after masks and prescales  - no seeds
    /// ////////////////////////////////////////////////////////////////////////////////
    if (!algoSeedResultMaskAndPresc)
      continue;

    /// Emulated GT result of algorithm is false - no seeds - but still save the event
    //  This should not happen if the emulated and unpacked GT are consistent
    /// ////////////////////////////////////////////////////////////////////////////////
    if (!algoSeedResult)
      continue;

    const std::vector<GlobalLogicParser::OperandToken>& opTokenVecObjMap = objMap->operandTokenVector();
    const std::vector<L1TObjectTypeInCond>& condObjTypeVec = objMap->objectTypeVector();
    const std::vector<CombinationsWithBxInCond>& condCombinations = objMap->combinationVector();

    LogTrace("HLTL1TSeed") << "\n\talgoName =" << objMap->algoName() << "\talgoBitNumber = " << algoSeedBitNumber
                           << "\talgoGtlResult = " << algoSeedResult << endl
                           << endl;

    if (opTokenVecObjMap.size() != condObjTypeVec.size()) {
      edm::LogWarning("HLTL1TSeed")
          << "\nWarning: GlobalObjectMapRecord with input tag " << m_l1GtObjectMapTag
          << "\nhas object map for bit number " << algoSeedBitNumber
          << " which contains different size vectors of operand tokens and of condition object types!" << std::endl;

      assert(opTokenVecObjMap.size() == condObjTypeVec.size());
    }

    if (opTokenVecObjMap.size() != condCombinations.size()) {
      edm::LogWarning("HLTL1TSeed")
          << "\nWarning: GlobalObjectMapRecord with input tag " << m_l1GtObjectMapTag
          << "\nhas object map for bit number " << algoSeedBitNumber
          << " which contains different size vectors of operand tokens and of condition object combinations!"
          << std::endl;

      assert(opTokenVecObjMap.size() == condCombinations.size());
    }

    // operands are conditions of L1 algo
    //
    for (size_t condNumber = 0; condNumber < opTokenVecObjMap.size(); condNumber++) {
      std::vector<l1t::GlobalObject> const& condObjType = condObjTypeVec[condNumber];

      for (auto const& jOb : condObjType) {
        LogTrace("HLTL1TSeed") << setw(15) << "\tcondObjType = " << jOb << endl;
      }

      bool const condResult = opTokenVecObjMap[condNumber].tokenResult;

      // only proceed for conditions that passed
      //
      if (!condResult) {
        continue;
      }

      // loop over combinations for a given condition
      //
      auto const& condComb = *(objMap->getCombinationsInCond(condNumber));

      LogTrace("HLTL1TSeed") << setw(15) << "\tcondCombinations = " << condComb.size();

      for (size_t i1 = 0; i1 < condComb.size(); ++i1) {
        LogTrace("HLTL1TSeed") << setw(15) << "\tnew combination" << endl;

        // loop over objects in a combination for a given condition
        //
        for (size_t i2 = 0; i2 < condComb[i1].size(); ++i2) {
          // in case of object-less triggers (e.g. L1_ZeroBias) condObjType vector is empty, so don't seed!
          //
          if (condObjType.empty()) {
            LogTrace("HLTL1TSeed")
                << "\talgoName = " << objMap->algoName()
                << " is object-less L1 algorithm, so do not attempt to store any objects to the list of seeds.\n"
                << std::endl;
            continue;
          }

          // BX of the L1T object
          auto const objBx = condComb[i1][i2].first;

          // index of the L1T object in the relevant BXVector
          auto const objIdx = condComb[i1][i2].second;

          // type of the L1T object
          // (the index of the object type is the same as the index of the object)
          l1t::GlobalObject const objTypeVal = condObjType[i2];

          LogTrace("HLTL1TSeed") << "\tAdd object of type " << objTypeVal << " and index " << objIdx
                                 << " (BX = " << objBx << ") to the seed list.";

          // THESE OBJECT CASES ARE CURRENTLY MISSING:
          //gtMinBias,
          //gtExternal,
          //ObjNull

          // fill list(s) of BX:index values
          switch (objTypeVal) {
            case l1t::gtMu: {
              listMuon.emplace_back(objBx, objIdx);
            } break;
            case l1t::gtMuShower: {
              listMuonShower.emplace_back(objBx, objIdx);
            } break;
            case l1t::gtEG: {
              listEG.emplace_back(objBx, objIdx);
            } break;
            case l1t::gtJet: {
              listJet.emplace_back(objBx, objIdx);
            } break;
            case l1t::gtTau: {
              listTau.emplace_back(objBx, objIdx);
            } break;
            case l1t::gtETM: {
              listETM.emplace_back(objBx, objIdx);
            } break;
            case l1t::gtETT: {
              listETT.emplace_back(objBx, objIdx);
            } break;
            case l1t::gtHTT: {
              listHTT.emplace_back(objBx, objIdx);
            } break;
            case l1t::gtHTM: {
              listHTM.emplace_back(objBx, objIdx);
            } break;
            case l1t::gtETMHF: {
              listETMHF.emplace_back(objBx, objIdx);
            } break;
            case l1t::gtHTMHF: {
              listHTMHF.emplace_back(objBx, objIdx);
            } break;
            case l1t::gtTowerCount: {
              listTowerCount.emplace_back(objBx, objIdx);
            } break;
            case l1t::gtMinBiasHFP0: {
              listMinBiasHFP0.emplace_back(objBx, objIdx);
            } break;
            case l1t::gtMinBiasHFM0: {
              listMinBiasHFM0.emplace_back(objBx, objIdx);
            } break;
            case l1t::gtMinBiasHFP1: {
              listMinBiasHFP1.emplace_back(objBx, objIdx);
            } break;
            case l1t::gtMinBiasHFM1: {
              listMinBiasHFM1.emplace_back(objBx, objIdx);
            } break;
            case l1t::gtETTem: {
              listTotalEtEm.emplace_back(objBx, objIdx);
            } break;
            case l1t::gtAsymmetryEt: {
              listAsymEt.emplace_back(objBx, objIdx);
            } break;
            case l1t::gtAsymmetryHt: {
              listAsymHt.emplace_back(objBx, objIdx);
            } break;
            case l1t::gtAsymmetryEtHF: {
              listAsymEtHF.emplace_back(objBx, objIdx);
            } break;
            case l1t::gtAsymmetryHtHF: {
              listAsymHtHF.emplace_back(objBx, objIdx);
            } break;
            case l1t::gtZDCP: {
              listZDCP.emplace_back(objBx, objIdx);
            } break;
            case l1t::gtZDCM: {
              listZDCM.emplace_back(objBx, objIdx);
            } break;
            case l1t::gtCentrality0:
            case l1t::gtCentrality1:
            case l1t::gtCentrality2:
            case l1t::gtCentrality3:
            case l1t::gtCentrality4:
            case l1t::gtCentrality5:
            case l1t::gtCentrality6:
            case l1t::gtCentrality7: {
              listCentrality.emplace_back(objBx, objIdx);
            } break;

            default: {
              // should not arrive here
              LogTrace("HLTL1TSeed") << "\n    HLTL1TSeed::hltFilter "
                                     << "\n      Unknown object of type " << objTypeVal << " and index " << objIdx
                                     << " (BX = " << objBx << ") in the seed list.";
            } break;

          }  // end switch objTypeVal

        }  // end for itObj

      }  // end for itComb

    }  // end for condition

  }  // end for itSeed

  // eliminate duplicates

  listMuon.sort();
  listMuon.unique();

  listMuonShower.sort();
  listMuonShower.unique();

  listEG.sort();
  listEG.unique();

  listJet.sort();
  listJet.unique();

  listTau.sort();
  listTau.unique();

  listETM.sort();
  listETM.unique();

  listETT.sort();
  listETT.unique();

  listHTT.sort();
  listHTT.unique();

  listHTM.sort();
  listHTM.unique();

  listETMHF.sort();
  listETMHF.unique();

  listHTMHF.sort();
  listHTMHF.unique();

  listCentrality.sort();
  listCentrality.unique();

  listMinBiasHFP0.sort();
  listMinBiasHFP0.unique();

  listMinBiasHFM0.sort();
  listMinBiasHFM0.unique();

  listMinBiasHFP1.sort();
  listMinBiasHFP1.unique();

  listMinBiasHFM1.sort();
  listMinBiasHFM1.unique();

  listTotalEtEm.sort();
  listTotalEtEm.unique();

  listMissingEtHF.sort();
  listMissingEtHF.unique();

  listTowerCount.sort();
  listTowerCount.unique();

  listAsymEt.sort();
  listAsymEt.unique();

  listAsymHt.sort();
  listAsymHt.unique();

  listAsymEtHF.sort();
  listAsymEtHF.unique();

  listAsymHtHF.sort();
  listAsymHtHF.unique();

  listZDCP.sort();
  listZDCP.unique();

  listZDCM.sort();
  listZDCM.unique();

  // record the L1 physics objects in the HLT filterproduct
  // //////////////////////////////////////////////////////

  // Muon
  if (!listMuon.empty()) {
    edm::Handle<l1t::MuonBxCollection> muons;
    iEvent.getByToken(m_l1MuonToken, muons);

    if (!muons.isValid()) {
      edm::LogWarning("HLTL1TSeed") << "\nWarning: L1MuonBxCollection with input tag " << m_l1MuonTag
                                    << "\nrequested in configuration, but not found in the event."
                                    << "\nNo muons added to filterproduct." << endl;
    } else {
      for (auto const& [bxIdx, objIdx] : listMuon) {
        // skip invalid indices
        if (objIdx < 0 or unsigned(objIdx) >= muons->size(bxIdx)) {
          edm::LogWarning("HLTL1TSeed")
              << "Invalid index from the L1ObjectMap (L1uGT emulator), will be ignored (l1t::MuonBxCollection):"
              << " index=" << objIdx << " (" << muons->size(bxIdx) << " unpacked L1T objects in BX " << bxIdx << ")";
          continue;
        }

        // Transform to index for Bx = 0 to begin of BxVector
        unsigned int index = muons->begin(bxIdx) - muons->begin() + objIdx;

        l1t::MuonRef myref(muons, index);
        filterproduct.addObject(trigger::TriggerL1Mu, myref);
      }
    }
  }

  // Muon Shower
  if (!listMuonShower.empty()) {
    edm::Handle<l1t::MuonShowerBxCollection> muonShowers;
    iEvent.getByToken(m_l1MuonShowerToken, muonShowers);

    if (!muonShowers.isValid()) {
      edm::LogWarning("HLTL1TSeed") << "\nWarning: L1MuonShowerBxCollection with input tag " << m_l1MuonShowerTag
                                    << "\nrequested in configuration, but not found in the event."
                                    << "\nNo muon showers added to filterproduct." << endl;
    } else {
      for (auto const& [bxIdx, objIdx] : listMuonShower) {
        // skip invalid indices
        if (objIdx < 0 or unsigned(objIdx) >= muonShowers->size(bxIdx)) {
          edm::LogWarning("HLTL1TSeed")
              << "Invalid index from the L1ObjectMap (L1uGT emulator), will be ignored (l1t::MuonShowerBxCollection):"
              << " index=" << objIdx << " (" << muonShowers->size(bxIdx) << " unpacked L1T objects in BX " << bxIdx
              << ")";
          continue;
        }

        // Transform to index for Bx = 0 to begin of BxVector
        unsigned int index = muonShowers->begin(bxIdx) - muonShowers->begin() + objIdx;

        l1t::MuonShowerRef myref(muonShowers, index);
        filterproduct.addObject(trigger::TriggerL1MuShower, myref);
      }
    }
  }

  // EG (isolated)
  if (!listEG.empty()) {
    edm::Handle<l1t::EGammaBxCollection> egammas;
    iEvent.getByToken(m_l1EGammaToken, egammas);
    if (!egammas.isValid()) {
      edm::LogWarning("HLTL1TSeed") << "\nWarning: L1EGammaBxCollection with input tag " << m_l1EGammaTag
                                    << "\nrequested in configuration, but not found in the event."
                                    << "\nNo egammas added to filterproduct." << endl;
    } else {
      for (auto const& [bxIdx, objIdx] : listEG) {
        // skip invalid indices
        if (objIdx < 0 or unsigned(objIdx) >= egammas->size(bxIdx)) {
          edm::LogWarning("HLTL1TSeed")
              << "Invalid index from the L1ObjectMap (L1uGT emulator), will be ignored (l1t::EGammaBxCollection):"
              << " index=" << objIdx << " (" << egammas->size(bxIdx) << " unpacked L1T objects in BX " << bxIdx << ")";
          continue;
        }

        // Transform to begin of BxVector
        unsigned int index = egammas->begin(bxIdx) - egammas->begin() + objIdx;

        l1t::EGammaRef myref(egammas, index);
        filterproduct.addObject(trigger::TriggerL1EG, myref);
      }
    }
  }

  // Jet
  if (!listJet.empty()) {
    edm::Handle<l1t::JetBxCollection> jets;
    iEvent.getByToken(m_l1JetToken, jets);

    if (!jets.isValid()) {
      edm::LogWarning("HLTL1TSeed") << "\nWarning: L1JetBxCollection with input tag " << m_l1JetTag
                                    << "\nrequested in configuration, but not found in the event."
                                    << "\nNo jets added to filterproduct." << endl;
    } else {
      for (auto const& [bxIdx, objIdx] : listJet) {
        // skip invalid indices
        if (objIdx < 0 or unsigned(objIdx) >= jets->size(bxIdx)) {
          edm::LogWarning("HLTL1TSeed")
              << "Invalid index from the L1ObjectMap (L1uGT emulator), will be ignored (l1t::JetBxCollection):"
              << " index=" << objIdx << " (" << jets->size(bxIdx) << " unpacked L1T objects in BX " << bxIdx << ")";
          continue;
        }

        // Transform to begin of BxVector
        unsigned int index = jets->begin(bxIdx) - jets->begin() + objIdx;

        l1t::JetRef myref(jets, index);
        filterproduct.addObject(trigger::TriggerL1Jet, myref);
      }
    }
  }

  // Tau
  if (!listTau.empty()) {
    edm::Handle<l1t::TauBxCollection> taus;
    iEvent.getByToken(m_l1TauToken, taus);

    if (!taus.isValid()) {
      edm::LogWarning("HLTL1TSeed") << "\nWarning: L1TauBxCollection with input tag " << m_l1TauTag
                                    << "\nrequested in configuration, but not found in the event."
                                    << "\nNo taus added to filterproduct." << endl;
    } else {
      for (auto const& [bxIdx, objIdx] : listTau) {
        // skip invalid indices
        if (objIdx < 0 or unsigned(objIdx) >= taus->size(bxIdx)) {
          edm::LogWarning("HLTL1TSeed")
              << "Invalid index from the L1ObjectMap (L1uGT emulator), will be ignored (l1t::TauBxCollection):"
              << " index=" << objIdx << " (" << taus->size(bxIdx) << " unpacked L1T objects in BX " << bxIdx << ")";
          continue;
        }

        // Transform to begin of BxVector
        unsigned int index = taus->begin(bxIdx) - taus->begin() + objIdx;

        l1t::TauRef myref(taus, index);
        filterproduct.addObject(trigger::TriggerL1Tau, myref);
      }
    }
  }

  // ETT, HTT, ETM, HTM
  auto fillEtSums = [&](edm::Handle<l1t::EtSumBxCollection> const& etSums,
                        idxListType const& theList,
                        l1t::EtSum::EtSumType const l1tId,
                        trigger::TriggerObjectType const trigObjId) {
    for (auto const& [bxIdx, objIdx] : theList) {
      for (auto iter = etSums->begin(bxIdx); iter != etSums->end(bxIdx); ++iter) {
        l1t::EtSumRef myref(etSums, etSums->key(iter));
        if (myref->getType() == l1tId) {
          filterproduct.addObject(trigObjId, myref);
        }
      }
    }
  };

  auto const etsums = iEvent.getHandle(m_l1EtSumToken);
  if (not etsums.isValid()) {
    edm::LogWarning("HLTL1TSeed") << "\nWarning: L1EtSumBxCollection with input tag " << m_l1EtSumTag
                                  << "\nrequested in configuration, but not found in the event."
                                  << "\nNo etsums added to filterproduct." << endl;
  } else {
    fillEtSums(etsums, listETT, l1t::EtSum::kTotalEt, trigger::TriggerL1ETT);
    fillEtSums(etsums, listHTT, l1t::EtSum::kTotalHt, trigger::TriggerL1HTT);
    fillEtSums(etsums, listETM, l1t::EtSum::kMissingEt, trigger::TriggerL1ETM);
    fillEtSums(etsums, listHTM, l1t::EtSum::kMissingHt, trigger::TriggerL1HTM);
    fillEtSums(etsums, listETMHF, l1t::EtSum::kMissingEtHF, trigger::TriggerL1ETMHF);
    fillEtSums(etsums, listHTMHF, l1t::EtSum::kMissingHtHF, trigger::TriggerL1HTMHF);
    fillEtSums(etsums, listCentrality, l1t::EtSum::kCentrality, trigger::TriggerL1Centrality);
    fillEtSums(etsums, listMinBiasHFP0, l1t::EtSum::kMinBiasHFP0, trigger::TriggerL1MinBiasHFP0);
    fillEtSums(etsums, listMinBiasHFM0, l1t::EtSum::kMinBiasHFM0, trigger::TriggerL1MinBiasHFM0);
    fillEtSums(etsums, listMinBiasHFP1, l1t::EtSum::kMinBiasHFP1, trigger::TriggerL1MinBiasHFP1);
    fillEtSums(etsums, listMinBiasHFM1, l1t::EtSum::kMinBiasHFM1, trigger::TriggerL1MinBiasHFM1);
    fillEtSums(etsums, listTotalEtEm, l1t::EtSum::kTotalEtEm, trigger::TriggerL1TotalEtEm);
    fillEtSums(etsums, listTowerCount, l1t::EtSum::kTowerCount, trigger::TriggerL1TowerCount);
    fillEtSums(etsums, listAsymEt, l1t::EtSum::kAsymEt, trigger::TriggerL1AsymEt);
    fillEtSums(etsums, listAsymHt, l1t::EtSum::kAsymHt, trigger::TriggerL1AsymHt);
    fillEtSums(etsums, listAsymEtHF, l1t::EtSum::kAsymEtHF, trigger::TriggerL1AsymEtHF);
    fillEtSums(etsums, listAsymHtHF, l1t::EtSum::kAsymHtHF, trigger::TriggerL1AsymHtHF);
  }

  // ZDCP, ZDCM
  auto const etsumzdcs = iEvent.getHandle(m_l1EtSumZdcToken);
  if (not etsumzdcs.isValid()) {
    edm::LogWarning("HLTL1TSeed") << "\nWarning: L1EtSumBxCollection with input tag " << m_l1EtSumZdcTag
                                  << "\nrequested in configuration, but not found in the event."
                                  << "\nNo etsums (ZDC) added to filterproduct.";
  } else {
    fillEtSums(etsumzdcs, listZDCP, l1t::EtSum::kZDCP, trigger::TriggerL1ZDCP);
    fillEtSums(etsumzdcs, listZDCM, l1t::EtSum::kZDCM, trigger::TriggerL1ZDCM);
  }

  // return filter decision
  LogTrace("HLTL1TSeed") << "\nHLTL1Seed:seedsL1TriggerObjectMaps returning " << seedsResult << endl << endl;

  return seedsResult;
}

// register as framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTL1TSeed);
