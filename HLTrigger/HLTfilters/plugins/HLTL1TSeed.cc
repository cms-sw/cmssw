#include <string>
#include <list>
#include <vector>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <cassert>

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
HLTL1TSeed::HLTL1TSeed(const edm::ParameterSet& parSet) : 
  HLTStreamFilter(parSet),
  //useObjectMaps_(parSet.getParameter<bool>("L1UseL1TriggerObjectMaps")),
  m_l1SeedsLogicalExpression(parSet.getParameter<string>("L1SeedsLogicalExpression")),
  m_l1GtObjectMapTag(parSet.getParameter<edm::InputTag> ("L1ObjectMapInputTag")),
  m_l1GtObjectMapToken(consumes<GlobalObjectMapRecord>(m_l1GtObjectMapTag)),
  m_l1GlobalTag(parSet.getParameter<edm::InputTag> ("L1GlobalInputTag")),
  m_l1GlobalToken(consumes<GlobalAlgBlkBxCollection>(m_l1GlobalTag)),
  m_l1MuonCollectionsTag(parSet.getParameter<edm::InputTag>("L1MuonInputTag")), // FIX WHEN UNPACKERS ADDED
  m_l1MuonTag(m_l1MuonCollectionsTag),
  m_l1MuonToken(consumes<l1t::MuonBxCollection>(m_l1MuonTag)),
  m_l1EGammaCollectionsTag(parSet.getParameter<edm::InputTag>("L1EGammaInputTag")), // FIX WHEN UNPACKERS ADDED
  m_l1EGammaTag(m_l1EGammaCollectionsTag),
  m_l1EGammaToken(consumes<l1t::EGammaBxCollection>(m_l1EGammaTag)),
  m_l1JetCollectionsTag(parSet.getParameter<edm::InputTag>("L1JetInputTag")), // FIX WHEN UNPACKERS ADDED
  m_l1JetTag(m_l1JetCollectionsTag),
  m_l1JetToken(consumes<l1t::JetBxCollection>(m_l1JetTag)),
  m_l1TauCollectionsTag(parSet.getParameter<edm::InputTag>("L1TauInputTag")), // FIX WHEN UNPACKERS ADDED
  m_l1TauTag(m_l1TauCollectionsTag),
  m_l1TauToken(consumes<l1t::TauBxCollection>(m_l1TauTag)),
  m_l1EtSumCollectionsTag(parSet.getParameter<edm::InputTag>("L1EtSumInputTag")), // FIX WHEN UNPACKERS ADDED
  m_l1EtSumTag(m_l1EtSumCollectionsTag),
  m_l1EtSumToken(consumes<l1t::EtSumBxCollection>(m_l1EtSumTag)),
  m_l1GlobalDecision(false),
  m_isDebugEnabled(edm::isDebugEnabled())
{

  if (m_l1SeedsLogicalExpression.empty()) {

    throw cms::Exception("FailModule") << "\nTrying to seed with an empty L1SeedsLogicalExpression.\n" << std::endl;

  }
  else if (m_l1SeedsLogicalExpression != "L1GlobalDecision") {

    // check also the logical expression - add/remove spaces if needed
    m_l1AlgoLogicParser = GlobalLogicParser(m_l1SeedsLogicalExpression);

    // list of required algorithms for seeding
    // dummy values for tokenNumber and tokenResult
    m_l1AlgoSeeds.reserve((m_l1AlgoLogicParser.operandTokenVector()).size());
    m_l1AlgoSeeds = m_l1AlgoLogicParser.expressionSeedsOperandList();
    size_t l1AlgoSeedsSize = m_l1AlgoSeeds.size();

    m_l1AlgoSeedsRpn.reserve(l1AlgoSeedsSize);
    m_l1AlgoSeedsObjType.reserve(l1AlgoSeedsSize);

  } 
  else {

    m_l1GlobalDecision = true;

  }

}

// destructor
HLTL1TSeed::~HLTL1TSeed() {
    // empty now
}

// member functions

void
HLTL1TSeed::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  makeHLTFilterDescription(desc);


  // # logical expression for the required L1 algorithms;
  // # the algorithms are specified by name
  // # allowed operators: "AND", "OR", "NOT", "(", ")"
  // #
  // # by convention, "L1GlobalDecision" logical expression means global decision
  desc.add<string>("L1SeedsLogicalExpression","");
  desc.add<edm::InputTag>("L1ObjectMapInputTag",edm::InputTag("hltGtStage2ObjectMap"));
  desc.add<edm::InputTag>("L1GlobalInputTag",edm::InputTag("hltGtStage2Digis"));
  desc.add<edm::InputTag>("L1MuonInputTag",edm::InputTag("hltGmtStage2Digis:Muon"));
  desc.add<edm::InputTag>("L1EGammaInputTag",edm::InputTag("hltCaloStage2Digis:EGamma"));
  desc.add<edm::InputTag>("L1JetInputTag",edm::InputTag("hltCaloStage2Digis:Jet"));
  desc.add<edm::InputTag>("L1TauInputTag",edm::InputTag("hltCaloStage2Digis:Tau"));
  desc.add<edm::InputTag>("L1EtSumInputTag",edm::InputTag("hltCaloStage2Digis:EtSum"));
  descriptions.add("hltL1TSeed", desc);
}

bool HLTL1TSeed::hltFilter(edm::Event& iEvent, const edm::EventSetup& evSetup, trigger::TriggerFilterObjectWithRefs & filterproduct) {

  bool rc = false;

  // the filter object
  if (saveTags()) {
    
    // muons
    filterproduct.addCollectionTag(m_l1MuonTag);

    // egamma 
    filterproduct.addCollectionTag(m_l1EGammaTag);

    // jet 
    filterproduct.addCollectionTag(m_l1JetTag);

    // tau 
    filterproduct.addCollectionTag(m_l1TauTag);

    // etsum 
    filterproduct.addCollectionTag(m_l1EtSumTag);
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
void HLTL1TSeed::dumpTriggerFilterObjectWithRefs(trigger::TriggerFilterObjectWithRefs & filterproduct) const
{

  LogTrace("HLTL1TSeed") 
  << "\nHLTL1TSeed::hltFilter "
  << "\n  Dump TriggerFilterObjectWithRefs\n" << endl;
  
  vector<l1t::MuonRef> seedsL1Mu;
  filterproduct.getObjects(trigger::TriggerL1Mu, seedsL1Mu);
  const size_t sizeSeedsL1Mu = seedsL1Mu.size();

  LogTrace("HLTL1TSeed")
  <<  "\n  HLTL1TSeed: seed logical expression = " << m_l1SeedsLogicalExpression << endl;

  LogTrace("HLTL1TSeed") 
  << "\n  L1Mu seeds:      " << sizeSeedsL1Mu << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1Mu; i++) {

    
    l1t::MuonRef obj = l1t::MuonRef( seedsL1Mu[i]);
    
        LogTrace("HLTL1TSeed") 
        << "\tL1Mu     " << "\t" << "q = " << obj->hwCharge()  // TEMP get hwCharge insead of charge which is not yet set NEED FIX.
        << "\t" << "pt = " << obj->pt() << "\t" << "eta =  " << obj->eta()
        << "\t" << "phi =  " << obj->phi();  //<< "\t" << "BX = " << obj->bx();
  }

  vector<l1t::EGammaRef> seedsL1EG;
  filterproduct.getObjects(trigger::TriggerL1EG, seedsL1EG);
  const size_t sizeSeedsL1EG = seedsL1EG.size();

  LogTrace("HLTL1TSeed") 
  << "\n  L1EG seeds:      " << sizeSeedsL1EG << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1EG; i++) {

    
    l1t::EGammaRef obj = l1t::EGammaRef( seedsL1EG[i]);
    
        LogTrace("HLTL1TSeed") 
        << "\tL1EG     " << "\t" << "pt = "
        << obj->pt() << "\t" << "eta =  " << obj->eta()
        << "\t" << "phi =  " << obj->phi();  //<< "\t" << "BX = " << obj->bx();
  }

  vector<l1t::JetRef> seedsL1Jet;
  filterproduct.getObjects(trigger::TriggerL1Jet, seedsL1Jet);
  const size_t sizeSeedsL1Jet = seedsL1Jet.size();

  LogTrace("HLTL1TSeed") 
  << "\n  L1Jet seeds:      " << sizeSeedsL1Jet << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1Jet; i++) {

    
    l1t::JetRef obj = l1t::JetRef( seedsL1Jet[i]);
    
        LogTrace("HLTL1TSeed") << "\tL1Jet     " << "\t" << "pt = "
        << obj->pt() << "\t" << "eta =  " << obj->eta()
        << "\t" << "phi =  " << obj->phi();  //<< "\t" << "BX = " << obj->bx();
  }

  vector<l1t::TauRef> seedsL1Tau;
  filterproduct.getObjects(trigger::TriggerL1Tau, seedsL1Tau);
  const size_t sizeSeedsL1Tau = seedsL1Tau.size();

  LogTrace("HLTL1TSeed") 
  << "\n  L1Tau seeds:      " << sizeSeedsL1Tau << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1Tau; i++) {

    
    l1t::TauRef obj = l1t::TauRef( seedsL1Tau[i]);
    
        LogTrace("HLTL1TSeed") 
        << "\tL1Tau     " << "\t" << "pt = "
        << obj->pt() << "\t" << "eta =  " << obj->eta()
        << "\t" << "phi =  " << obj->phi();  //<< "\t" << "BX = " << obj->bx();
  }

  vector<l1t::EtSumRef> seedsL1EtSumETT;
  filterproduct.getObjects(trigger::TriggerL1ETT, seedsL1EtSumETT);
  const size_t sizeSeedsL1EtSumETT = seedsL1EtSumETT.size();
  LogTrace("HLTL1TSeed") 
  << "\n  L1EtSum ETT seeds:      " << sizeSeedsL1EtSumETT << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1EtSumETT; i++) { 
    l1t::EtSumRef obj = l1t::EtSumRef( seedsL1EtSumETT[i]);
    
    LogTrace("HLTL1TSeed") 
    << "\tL1EtSum  ETT" << "\t" << "pt = "
    << obj->pt() << "\t" << "eta =  " << obj->eta()
    << "\t" << "phi =  " << obj->phi();  //<< "\t" << "BX = " << obj->bx();
  }

  vector<l1t::EtSumRef> seedsL1EtSumHTT;
  filterproduct.getObjects(trigger::TriggerL1HTT, seedsL1EtSumHTT);
  const size_t sizeSeedsL1EtSumHTT = seedsL1EtSumHTT.size();
  LogTrace("HLTL1TSeed") 
  << "\n  L1EtSum HTT seeds:      " << sizeSeedsL1EtSumHTT << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1EtSumHTT; i++) { 
    l1t::EtSumRef obj = l1t::EtSumRef( seedsL1EtSumHTT[i]);
    
    LogTrace("HLTL1TSeed") 
    << "\tL1EtSum  HTT" << "\t" << "pt = "
    << obj->pt() << "\t" << "eta =  " << obj->eta()
    << "\t" << "phi =  " << obj->phi();  //<< "\t" << "BX = " << obj->bx();
  }

  vector<l1t::EtSumRef> seedsL1EtSumETM;
  filterproduct.getObjects(trigger::TriggerL1ETM, seedsL1EtSumETM);
  const size_t sizeSeedsL1EtSumETM = seedsL1EtSumETM.size();
  LogTrace("HLTL1TSeed") 
  << "\n  L1EtSum ETM seeds:      " << sizeSeedsL1EtSumETM << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1EtSumETM; i++) { 
    l1t::EtSumRef obj = l1t::EtSumRef( seedsL1EtSumETM[i]);
    
    LogTrace("HLTL1TSeed") 
    << "\tL1EtSum  ETM" << "\t" << "pt = "
    << obj->pt() << "\t" << "eta =  " << obj->eta()
    << "\t" << "phi =  " << obj->phi();  //<< "\t" << "BX = " << obj->bx();
  }

  vector<l1t::EtSumRef> seedsL1EtSumETMHF;
  filterproduct.getObjects(trigger::TriggerL1ETMHF, seedsL1EtSumETMHF);
  const size_t sizeSeedsL1EtSumETMHF = seedsL1EtSumETMHF.size();
  LogTrace("HLTL1TSeed") 
  << "\n  L1EtSum ETMHF seeds:      " << sizeSeedsL1EtSumETMHF << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1EtSumETMHF; i++) { 
    l1t::EtSumRef obj = l1t::EtSumRef( seedsL1EtSumETMHF[i]);
    
    LogTrace("HLTL1TSeed") 
    << "\tL1EtSum  ETMHF" << "\t" << "pt = "
    << obj->pt() << "\t" << "eta =  " << obj->eta()
    << "\t" << "phi =  " << obj->phi();  //<< "\t" << "BX = " << obj->bx();
  }

  vector<l1t::EtSumRef> seedsL1EtSumHTM;
  filterproduct.getObjects(trigger::TriggerL1HTM, seedsL1EtSumHTM);
  const size_t sizeSeedsL1EtSumHTM = seedsL1EtSumHTM.size();
  LogTrace("HLTL1TSeed") 
  << "\n  L1EtSum HTM seeds:      " << sizeSeedsL1EtSumHTM << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1EtSumHTM; i++) { 
    l1t::EtSumRef obj = l1t::EtSumRef( seedsL1EtSumHTM[i]);
    
    LogTrace("HLTL1TSeed") 
    << "\tL1EtSum  HTM" << "\t" << "pt = "
    << obj->pt() << "\t" << "eta =  " << obj->eta()
    << "\t" << "phi =  " << obj->phi();  //<< "\t" << "BX = " << obj->bx();
  }

  vector<l1t::EtSumRef> seedsL1EtSumCentrality;
  filterproduct.getObjects(trigger::TriggerL1Centrality, seedsL1EtSumCentrality);
  const size_t sizeSeedsL1EtSumCentrality = seedsL1EtSumCentrality.size();
  LogTrace("HLTL1TSeed") 
  << "\n  L1EtSum Centrality seeds:      " << sizeSeedsL1EtSumCentrality << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1EtSumCentrality; i++) { 
    l1t::EtSumRef obj = l1t::EtSumRef( seedsL1EtSumCentrality[i]);
    
    LogTrace("HLTL1TSeed") 
    << "\tL1EtSum  Centrality Bits: " << std::bitset<8>(obj->hwPt()) << " (hwPt = " << obj->hwPt() << ")" ;
  }

  vector<l1t::EtSumRef> seedsL1EtSumMinBiasHFP0;
  filterproduct.getObjects(trigger::TriggerL1MinBiasHFP0, seedsL1EtSumMinBiasHFP0);
  const size_t sizeSeedsL1EtSumMinBiasHFP0 = seedsL1EtSumMinBiasHFP0.size();
  LogTrace("HLTL1TSeed") 
  << "\n  L1EtSum MinBiasHFP0 seeds:      " << sizeSeedsL1EtSumMinBiasHFP0 << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1EtSumMinBiasHFP0; i++) { 
    l1t::EtSumRef obj = l1t::EtSumRef( seedsL1EtSumMinBiasHFP0[i]);
    
    LogTrace("HLTL1TSeed") 
    << "\tL1EtSum  MinBiasHFP0: hwPt = " << obj->hwPt();
  }

  vector<l1t::EtSumRef> seedsL1EtSumMinBiasHFM0;
  filterproduct.getObjects(trigger::TriggerL1MinBiasHFM0, seedsL1EtSumMinBiasHFM0);
  const size_t sizeSeedsL1EtSumMinBiasHFM0 = seedsL1EtSumMinBiasHFM0.size();
  LogTrace("HLTL1TSeed") 
  << "\n  L1EtSum MinBiasHFM0 seeds:      " << sizeSeedsL1EtSumMinBiasHFM0 << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1EtSumMinBiasHFM0; i++) { 
    l1t::EtSumRef obj = l1t::EtSumRef( seedsL1EtSumMinBiasHFM0[i]);
    
    LogTrace("HLTL1TSeed") 
    << "\tL1EtSum  MinBiasHFM0: hwPt = " << obj->hwPt();
  }

  vector<l1t::EtSumRef> seedsL1EtSumMinBiasHFP1;
  filterproduct.getObjects(trigger::TriggerL1MinBiasHFP1, seedsL1EtSumMinBiasHFP1);
  const size_t sizeSeedsL1EtSumMinBiasHFP1 = seedsL1EtSumMinBiasHFP1.size();
  LogTrace("HLTL1TSeed") 
  << "\n  L1EtSum MinBiasHFP1 seeds:      " << sizeSeedsL1EtSumMinBiasHFP1 << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1EtSumMinBiasHFP1; i++) { 
    l1t::EtSumRef obj = l1t::EtSumRef( seedsL1EtSumMinBiasHFP1[i]);
    
    LogTrace("HLTL1TSeed") 
    << "\tL1EtSum  MinBiasHFP1: hwPt = " << obj->hwPt();
  }

  vector<l1t::EtSumRef> seedsL1EtSumMinBiasHFM1;
  filterproduct.getObjects(trigger::TriggerL1MinBiasHFM1, seedsL1EtSumMinBiasHFM1);
  const size_t sizeSeedsL1EtSumMinBiasHFM1 = seedsL1EtSumMinBiasHFM1.size();
  LogTrace("HLTL1TSeed") 
  << "\n  L1EtSum MinBiasHFM1 seeds:      " << sizeSeedsL1EtSumMinBiasHFM1 << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1EtSumMinBiasHFM1; i++) { 
    l1t::EtSumRef obj = l1t::EtSumRef( seedsL1EtSumMinBiasHFM1[i]);
    
    LogTrace("HLTL1TSeed") 
    << "\tL1EtSum  MinBiasHFM1: hwPt = " << obj->hwPt();
  }

  vector<l1t::EtSumRef> seedsL1EtSumTowerCount;
  filterproduct.getObjects(trigger::TriggerL1TowerCount, seedsL1EtSumTowerCount);
  const size_t sizeSeedsL1EtSumTowerCount = seedsL1EtSumTowerCount.size();
  LogTrace("HLTL1TSeed") 
  << "\n  L1EtSum TowerCount seeds:      " << sizeSeedsL1EtSumTowerCount << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1EtSumTowerCount; i++) { 
    l1t::EtSumRef obj = l1t::EtSumRef( seedsL1EtSumTowerCount[i]);
    
    LogTrace("HLTL1TSeed") 
    << "\tL1EtSum  TowerCount: hwPt = " << obj->hwPt();
  }

  vector<l1t::EtSumRef> seedsL1EtSumAsymEt;
  filterproduct.getObjects(trigger::TriggerL1AsymEt, seedsL1EtSumAsymEt);
  const size_t sizeSeedsL1EtSumAsymEt = seedsL1EtSumAsymEt.size();
  LogTrace("HLTL1TSeed") 
  << "\n  L1EtSum AsymEt seeds:      " << sizeSeedsL1EtSumAsymEt << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1EtSumAsymEt; i++) { 
    l1t::EtSumRef obj = l1t::EtSumRef( seedsL1EtSumAsymEt[i]);
    
    LogTrace("HLTL1TSeed") 
    << "\tL1EtSum  AsymEt: hwPt = " << obj->hwPt();
  }

  vector<l1t::EtSumRef> seedsL1EtSumAsymHt;
  filterproduct.getObjects(trigger::TriggerL1AsymHt, seedsL1EtSumAsymHt);
  const size_t sizeSeedsL1EtSumAsymHt = seedsL1EtSumAsymHt.size();
  LogTrace("HLTL1TSeed") 
  << "\n  L1EtSum AsymHt seeds:      " << sizeSeedsL1EtSumAsymHt << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1EtSumAsymHt; i++) { 
    l1t::EtSumRef obj = l1t::EtSumRef( seedsL1EtSumAsymHt[i]);
    
    LogTrace("HLTL1TSeed") 
    << "\tL1EtSum  AsymHt: hwPt = " << obj->hwPt();
  }

  vector<l1t::EtSumRef> seedsL1EtSumAsymEtHF;
  filterproduct.getObjects(trigger::TriggerL1AsymEtHF, seedsL1EtSumAsymEtHF);
  const size_t sizeSeedsL1EtSumAsymEtHF = seedsL1EtSumAsymEtHF.size();
  LogTrace("HLTL1TSeed") 
  << "\n  L1EtSum AsymEtHF seeds:      " << sizeSeedsL1EtSumAsymEtHF << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1EtSumAsymEtHF; i++) { 
    l1t::EtSumRef obj = l1t::EtSumRef( seedsL1EtSumAsymEtHF[i]);
    
    LogTrace("HLTL1TSeed") 
    << "\tL1EtSum  AsymEtHF: hwPt = " << obj->hwPt();
  }

  vector<l1t::EtSumRef> seedsL1EtSumAsymHtHF;
  filterproduct.getObjects(trigger::TriggerL1AsymHtHF, seedsL1EtSumAsymHtHF);
  const size_t sizeSeedsL1EtSumAsymHtHF = seedsL1EtSumAsymHtHF.size();
  LogTrace("HLTL1TSeed") 
  << "\n  L1EtSum AsymHtHF seeds:      " << sizeSeedsL1EtSumAsymHtHF << endl << endl;

  for (size_t i = 0; i != sizeSeedsL1EtSumAsymHtHF; i++) { 
    l1t::EtSumRef obj = l1t::EtSumRef( seedsL1EtSumAsymHtHF[i]);
    
    LogTrace("HLTL1TSeed") 
    << "\tL1EtSum  AsymHtHF: hwPt = " << obj->hwPt();
  }

  LogTrace("HLTL1TSeed") << " \n\n" << endl;

}

// seeding is done via L1 trigger object maps, considering the objects which fired in L1
bool HLTL1TSeed::seedsL1TriggerObjectMaps(edm::Event& iEvent,
        trigger::TriggerFilterObjectWithRefs & filterproduct
        ) {
    
    // Two GT objects are obtained from the Event: (1) the unpacked GT and (2) the emulated GT.
    // Return value of the function is the score of seeding logical expression, evaluated using (1).
    // Seeding is performed (per l1_algo) if ACCEPT both in (1) and (2). Seed objects are identified 
    // and only available from ObjectMaps created in (2).


    // define index lists for all particle types

    std::list<int> listMuon;

    std::list<int> listEG;

    std::list<int> listJet;
    std::list<int> listTau;

    std::list<int> listETM;
    std::list<int> listETT;
    std::list<int> listHTT;
    std::list<int> listHTM;
    std::list<int> listETMHF;

    std::list<int> listJetCounts;

    std::list<int> listCentrality;
    std::list<int> listMinBiasHFP0;
    std::list<int> listMinBiasHFM0;
    std::list<int> listMinBiasHFP1;
    std::list<int> listMinBiasHFM1;
    std::list<int> listTotalEtEm;
    std::list<int> listMissingEtHF;
    std::list<int> listTowerCount;
    std::list<int> listAsymEt;
    std::list<int> listAsymHt;
    std::list<int> listAsymEtHF;
    std::list<int> listAsymHtHF;

    // get handle to unpacked GT
    edm::Handle<GlobalAlgBlkBxCollection> uGtAlgoBlocks;
    iEvent.getByToken(m_l1GlobalToken, uGtAlgoBlocks);

    if (!uGtAlgoBlocks.isValid()) {

      edm::LogWarning("HLTL1TSeed")
      << " Warning: GlobalAlgBlkBxCollection with input tag "
      << m_l1GlobalTag
      << " requested in configuration, but not found in the event." << std::endl;

      return false;
    }

    // check size
    if(uGtAlgoBlocks->size() == 0) {

      edm::LogWarning("HLTL1TSeed")
      << " Warning: GlobalAlgBlkBxCollection with input tag "
      << m_l1GlobalTag
      << " is empty." << std::endl;

      return false;
    }

    // get handle to object maps from emulator (one object map per algorithm)
    edm::Handle<GlobalObjectMapRecord> gtObjectMapRecord;
    iEvent.getByToken(m_l1GtObjectMapToken, gtObjectMapRecord);

    if (!gtObjectMapRecord.isValid()) {

        edm::LogWarning("HLTL1TSeed")
        << " Warning: GlobalObjectMapRecord with input tag "
        << m_l1GtObjectMapTag
        << " requested in configuration, but not found in the event." << std::endl;

        return false;
    }

    if (m_isDebugEnabled) {

      const std::vector<GlobalObjectMap>& objMaps = gtObjectMapRecord->gtObjectMap();

      LogTrace("HLTL1TSeed") 
      << "\nHLTL1Seed"  
      << "\n---------------------------------------------------------------------------------------------------------------------";

      LogTrace("HLTL1TSeed") 
      << "\n\tAlgorithms in L1TriggerObjectMapRecord and GT results ( emulated | initial | prescaled | final ) " << endl;

      LogTrace("HLTL1TSeed") 
      << "\n\tmap" <<"\tAlgoBit" << std::setw(40) << "algoName" << "\t (emul|ini|pre|fin)" << endl;

      LogTrace("HLTL1TSeed")
      << "---------------------------------------------------------------------------------------------------------------------";

      for (size_t imap =0; imap < objMaps.size(); imap++) {

        int bit = objMaps[imap].algoBitNumber();   //  same as bit from L1T Menu

        int emulDecision = objMaps[imap].algoGtlResult();

        // For bx=0 , get 0th AlgoBlock, so in BXvector at(bx=0,i=0)
        int initDecision = (uGtAlgoBlocks->at(0,0)).getAlgoDecisionInitial(bit);
        int presDecision = (uGtAlgoBlocks->at(0,0)).getAlgoDecisionInterm(bit);
        int finlDecision = (uGtAlgoBlocks->at(0,0)).getAlgoDecisionFinal(bit);

        if(emulDecision != initDecision) {

          LogTrace("HLTL1TSeed") 
          << "L1T decision (emulated vs. unpacked initial) is not the same:"
          << "\n\tbit = " << std::setw(3) << bit 
          << std::setw(40) << objMaps[imap].algoName() 
          << "\t emulated decision = " << emulDecision << "\t unpacked initial decision = " << initDecision
          << "\nThis should not happen. Include the L1TGtEmulCompare module in the sequence."<< endl;

        }
        

        LogTrace("HLTL1TSeed")
        << "\t" << std::setw(3) << imap 
        << "\tbit = " << std::setw(3) << bit 
        << std::setw(40) << objMaps[imap].algoName() 
        << "\t (  " << emulDecision << " | " << initDecision << " | " << presDecision << " | " << finlDecision << " ) ";
        

      }
      LogTrace("HLTL1TSeed") << endl;

    }

    // Filter decision in case of "L1GlobalDecision" logical expression.
    // By convention, it means global decision.
    // /////////////////////////////////////////////////////////////////
    if (m_l1GlobalDecision) {

      // For bx=0 , get 0th AlgoBlock, so in BXvector at(bx=0,i=0)
      return (uGtAlgoBlocks->at(0,0)).getFinalOR();

    }


    // Update/Reset m_l1AlgoLogicParser by reseting token result 
    // /////////////////////////////////////////////////////////
    std::vector<GlobalLogicParser::OperandToken>& algOpTokenVector =
            m_l1AlgoLogicParser.operandTokenVector();

    for (auto & i : algOpTokenVector) {

        // rest token result 
        //
        i.tokenResult = false;

    }

    // Update m_l1AlgoLogicParser and store emulator results for algOpTokens 
    // /////////////////////////////////////////////////////////////////////
    for (auto & i : algOpTokenVector) {

        std::string algoName = i.tokenName;

        const GlobalObjectMap* objMap = gtObjectMapRecord->getObjectMap(algoName);

        if(objMap == nullptr) {

          throw cms::Exception("FailModule") << "\nAlgorithm " << algoName 
            << ", requested as seed by a HLT path, cannot be matched to a L1 algo name in any GlobalObjectMap\n" 
            << "Please check if algorithm " << algoName << " is present in the L1 menu\n" << std::endl;

        }
        else {

          //(algOpTokenVector[i]).tokenResult = objMap->algoGtlResult();

          int bit = objMap->algoBitNumber();
          bool finalAlgoDecision = (uGtAlgoBlocks->at(0,0)).getAlgoDecisionFinal(bit);
          i.tokenResult = finalAlgoDecision;

        }

    }

    // Filter decision 
    // ///////////////
    bool seedsResult = m_l1AlgoLogicParser.expressionResult();

    if (m_isDebugEnabled ) {

        LogTrace("HLTL1TSeed")
        << "\nHLTL1TSeed: l1SeedsLogicalExpression (names) = '" << m_l1SeedsLogicalExpression << "'"
        << "\n  Result for logical expression after update of algOpTokens: " << seedsResult << "\n"
        << std::endl;
    }

    /// Loop over the list of required algorithms for seeding
    /// /////////////////////////////////////////////////////

    for (std::vector<GlobalLogicParser::OperandToken>::const_iterator
            itSeed = m_l1AlgoSeeds.begin(); itSeed != m_l1AlgoSeeds.end(); ++itSeed) {
      
      std::string algoSeedName = (*itSeed).tokenName;

      LogTrace("HLTL1TSeed") 
      << "\n ----------------  algo seed name = " << algoSeedName << endl;

      const GlobalObjectMap* objMap = gtObjectMapRecord->getObjectMap(algoSeedName);

      if(objMap == nullptr) {

          // Should not get here
          //
          throw cms::Exception("FailModule") << "\nAlgorithm " << algoSeedName 
            << ", requested as seed by a HLT path, cannot be matched to a L1 algo name in any GlobalObjectMap\n" 
            << "Please check if algorithm " << algoSeedName << " is present in the L1 menu\n" << std::endl;

      }

      int  algoSeedBitNumber = objMap->algoBitNumber();
      bool algoSeedResult    = objMap->algoGtlResult();

      // unpacked GT results: uGtAlgoBlock has decisions initial, prescaled, and final after masks
      bool algoSeedResultMaskAndPresc = uGtAlgoBlocks->at(0,0).getAlgoDecisionFinal(algoSeedBitNumber); 

      LogTrace("HLTL1TSeed") 
      << "\n\tAlgo seed " << algoSeedName << " result emulated | final = " << algoSeedResult  << " | " << algoSeedResultMaskAndPresc << endl;

      /// Unpacked GT result of algorithm is false after masks and prescales  - no seeds
      /// ////////////////////////////////////////////////////////////////////////////////
      if(!algoSeedResultMaskAndPresc) continue;

      /// Emulated GT result of algorithm is false - no seeds - but still save the event
      //  This should not happen if the emulated and unpacked GT are consistent
      /// ////////////////////////////////////////////////////////////////////////////////
      if(!algoSeedResult) continue; 

      const std::vector<GlobalLogicParser::OperandToken>& opTokenVecObjMap = objMap->operandTokenVector();
      const std::vector<L1TObjectTypeInCond>&  condObjTypeVec = objMap->objectTypeVector();
      const std::vector<CombinationsInCond>& condCombinations = objMap->combinationVector();

      LogTrace("HLTL1TSeed")
      << "\n\talgoName =" << objMap->algoName() 
      << "\talgoBitNumber = " << algoSeedBitNumber 
      << "\talgoGtlResult = " << algoSeedResult << endl << endl;


      if (opTokenVecObjMap.size() != condObjTypeVec.size() ) {
          edm::LogWarning("HLTL1TSeed")
          << "\nWarning: GlobalObjectMapRecord with input tag "
          << m_l1GtObjectMapTag
          << "\nhas object map for bit number " << algoSeedBitNumber << " which contains different size vectors of operand tokens and of condition object types!"  << std::endl;
    
          assert(opTokenVecObjMap.size() == condObjTypeVec.size());
      }

      if (opTokenVecObjMap.size() != condCombinations.size()) {
          edm::LogWarning("HLTL1TSeed")
          << "\nWarning: GlobalObjectMapRecord with input tag "
          << m_l1GtObjectMapTag
          << "\nhas object map for bit number " << algoSeedBitNumber << " which contains different size vectors of operand tokens and of condition object combinations!"  << std::endl;
    
          assert(opTokenVecObjMap.size() == condCombinations.size());
      }

      // operands are conditions of L1 algo
      //
      for (size_t condNumber = 0; condNumber < opTokenVecObjMap.size(); condNumber++) {

        std::vector<l1t::GlobalObject> condObjType = condObjTypeVec[condNumber];

        for (auto & jOb : condObjType) {

          LogTrace("HLTL1TSeed")
          << setw(15) << "\tcondObjType = " << jOb << endl;

        }

        const std::string condName = opTokenVecObjMap[condNumber].tokenName;
        bool condResult = opTokenVecObjMap[condNumber].tokenResult;

        // only procede for conditions that passed
        //
        if ( !condResult) {
            continue;
        }

        // loop over combinations for a given condition
        //
        const CombinationsInCond* condComb = objMap->getCombinationsInCond(condNumber);

        LogTrace("HLTL1TSeed")
        << setw(15) << "\tcondCombinations = " << condComb->size() << endl;

        for (auto const & itComb : (*condComb)) {

            LogTrace("HLTL1TSeed")
            << setw(15) << "\tnew combination" << endl;

            // loop over objects in a combination for a given condition
            //
            for (auto itObject = itComb.begin(); itObject != itComb.end(); itObject++) {

                // in case of object-less triggers (e.g. L1_ZeroBias) condObjType vector is empty, so don't seed!
                //
                if(condObjType.empty()) {

                  LogTrace("HLTL1TSeed")
                  << "\talgoName = " << objMap->algoName() << " is object-less L1 algorithm, so do not attempt to store any objects to the list of seeds.\n"
                  << std::endl;
                  continue;

                }

                // the index of the object type is the same as the index of the object
                size_t iType = std::distance(itComb.begin(), itObject);

                // get object type and push indices on the list
                //
                const l1t::GlobalObject objTypeVal = condObjType.at(iType);

                LogTrace("HLTL1TSeed")
                << "\tAdd object of type " << objTypeVal << " and index " << (*itObject) << " to the seed list."
                << std::endl;


		// THESE OBJECT CASES ARE CURRENTLY MISSING:
		//gtMinBias,
		//gtExternal,
		//ObjNull

                switch (objTypeVal) {
		    case l1t::gtMu: {
                        listMuon.push_back(*itObject);
                    }

                    break;
                    case l1t::gtEG: {
                        listEG.push_back(*itObject);
                    }
		    break;
                    case l1t::gtJet: {
                        listJet.push_back(*itObject);
                    }
                    break;
                    case l1t::gtTau: {
                        listTau.push_back(*itObject);
                    }
                    break;
                    case l1t::gtETM: {
                        listETM.push_back(*itObject);
                    }
                    break;
                    case l1t::gtETT: {
                        listETT.push_back(*itObject);
                    }
                    break;
                    case l1t::gtHTT: {
                        listHTT.push_back(*itObject);
                    }
                    break;
                    case l1t::gtHTM: {
                        listHTM.push_back(*itObject);
                    }
                    break;
                    case l1t::gtETMHF: {
                        listETMHF.push_back(*itObject);
                    }
                    break;
                    case l1t::gtTowerCount: {
                        listTowerCount.push_back(*itObject);
                    }
                    break;
                    case l1t::gtMinBiasHFP0: {
                        listMinBiasHFP0.push_back(*itObject);
                    }
                    break;
                    case l1t::gtMinBiasHFM0: {
                        listMinBiasHFM0.push_back(*itObject);
                    }
                    break;
                    case l1t::gtMinBiasHFP1: {
                        listMinBiasHFP1.push_back(*itObject);
                    }
                    break;
                    case l1t::gtMinBiasHFM1: {
                        listMinBiasHFM1.push_back(*itObject);
                    }
                    break;
                    case l1t::gtETTem: {
                        listTotalEtEm.push_back(*itObject);
                    }
                    break;
                    case l1t::gtAsymmetryEt: {
                        listAsymEt.push_back(*itObject);
                    }
                    break;
                    case l1t::gtAsymmetryHt: {
                        listAsymHt.push_back(*itObject);
                    }
                    break;
                    case l1t::gtAsymmetryEtHF: {
                        listAsymEtHF.push_back(*itObject);
                    }
                    break;
                    case l1t::gtAsymmetryHtHF: {
                        listAsymHtHF.push_back(*itObject);
                    }
                    break;
                    case l1t::gtCentrality0: 
                    case l1t::gtCentrality1: 
                    case l1t::gtCentrality2: 
                    case l1t::gtCentrality3: 
                    case l1t::gtCentrality4: 
                    case l1t::gtCentrality5: 
                    case l1t::gtCentrality6: 
                    case l1t::gtCentrality7: {
                        listCentrality.push_back(*itObject);
                    }
                    break;

                    //case JetCounts: {
                    //    listJetCounts.push_back(*itObject);
                    //}

                    break;
                    default: {
                        // should not arrive here

                        LogTrace("HLTL1TSeed")
                        << "\n    HLTL1TSeed::hltFilter "
                        << "\n      Unknown object of type " << objTypeVal
                        << " and index " << (*itObject) << " in the seed list."
                        << std::endl;
                    }
                    break;

                } // end switch objTypeVal

          } // end for itObj

        } // end for itComb

      } // end for condition

    } // end for itSeed


    // eliminate duplicates

    listMuon.sort();
    listMuon.unique();

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

    listJetCounts.sort();
    listJetCounts.unique();

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
    
    // record the L1 physics objects in the HLT filterproduct
    // //////////////////////////////////////////////////////

    // Muon
    if (!listMuon.empty()) {

      edm::Handle<l1t::MuonBxCollection> muons;
      iEvent.getByToken(m_l1MuonToken, muons);

      if (!muons.isValid()){ 
          edm::LogWarning("HLTL1TSeed")
    	  << "\nWarning: L1MuonBxCollection with input tag "
    	  << m_l1MuonTag
    	  << "\nrequested in configuration, but not found in the event."
    	  << "\nNo muons added to filterproduct."
    	  << endl;	
      } 
      else {
    
        for (std::list<int>::const_iterator itObj = listMuon.begin(); itObj != listMuon.end(); ++itObj) {
    	

          // Transform to index for Bx = 0 to begin of BxVector
          unsigned int index = muons->begin(0) - muons->begin() + *itObj;

    	  l1t::MuonRef myref(muons, index);
    	  filterproduct.addObject(trigger::TriggerL1Mu, myref);

        }

      } 

    }

    // EG (isolated)
    if (!listEG.empty()) {

      edm::Handle<l1t::EGammaBxCollection> egammas;
      iEvent.getByToken(m_l1EGammaToken, egammas);
      if (!egammas.isValid()){ 
        edm::LogWarning("HLTL1TSeed")
        << "\nWarning: L1EGammaBxCollection with input tag " << m_l1EGammaTag
        << "\nrequested in configuration, but not found in the event."
        << "\nNo egammas added to filterproduct."
        << endl;	
      } 
      else {
    
        for (std::list<int>::const_iterator itObj = listEG.begin(); itObj != listEG.end(); ++itObj) {

          // Transform to begin of BxVector
          unsigned int index = egammas->begin(0) - egammas->begin() + *itObj;

    	  l1t::EGammaRef myref(egammas, index);
    	  filterproduct.addObject(trigger::TriggerL1EG, myref);

        } 

      } 
    
    } 

    // Jet
    if (!listJet.empty()) {
    
      edm::Handle<l1t::JetBxCollection> jets;
      iEvent.getByToken(m_l1JetToken, jets);

      if (!jets.isValid()){ 
        edm::LogWarning("HLTL1TSeed")
        << "\nWarning: L1JetBxCollection with input tag " << m_l1JetTag
        << "\nrequested in configuration, but not found in the event."
        << "\nNo jets added to filterproduct."
        << endl;	
      } 
      else {
  
        for (std::list<int>::const_iterator itObj = listJet.begin(); itObj != listJet.end(); ++itObj) {

          // Transform to begin of BxVector
          unsigned int index = jets->begin(0) - jets->begin() + *itObj;

          l1t::JetRef myref(jets, index);
          filterproduct.addObject(trigger::TriggerL1Jet, myref); 

        }

      }

    }

    // Tau
    if (!listTau.empty()) {
    
      edm::Handle<l1t::TauBxCollection> taus;
      iEvent.getByToken(m_l1TauToken, taus);

      if (!taus.isValid()){ 
        edm::LogWarning("HLTL1TSeed")
        << "\nWarning: L1TauBxCollection with input tag " << m_l1TauTag
        << "\nrequested in configuration, but not found in the event."
        << "\nNo taus added to filterproduct."
        << endl;	
      } 
      else {
  
        for (std::list<int>::const_iterator itObj = listTau.begin(); itObj != listTau.end(); ++itObj) {

          // Transform to begin of BxVector
          unsigned int index = taus->begin(0) - taus->begin() + *itObj;

          l1t::TauRef myref(taus, index);
          filterproduct.addObject(trigger::TriggerL1Tau, myref); 

        }

      }

    }

    // ETT, HTT, ETM, HTM
		edm::Handle<l1t::EtSumBxCollection> etsums;
		iEvent.getByToken(m_l1EtSumToken, etsums);
		if (!etsums.isValid()){ 
		  edm::LogWarning("HLTL1TSeed")
		    << "\nWarning: L1EtSumBxCollection with input tag "
		    << m_l1EtSumTag
		    << "\nrequested in configuration, but not found in the event."
		    << "\nNo etsums added to filterproduct."
		    << endl;	
		} else {
		  
			l1t::EtSumBxCollection::const_iterator iter;
			
			for (iter = etsums->begin(0); iter != etsums->end(0); ++iter){
			
			  l1t::EtSumRef myref(etsums, etsums->key(iter));
			
			  switch(iter->getType()) {

			    case l1t::EtSum::kTotalEt : 
            if(!listETT.empty())
			        filterproduct.addObject(trigger::TriggerL1ETT, myref); 
			      break;
			    case l1t::EtSum::kTotalHt : 
            if(!listHTT.empty())
			        filterproduct.addObject(trigger::TriggerL1HTT, myref); 
			      break;
			    case l1t::EtSum::kMissingEt: 
            if(!listETM.empty())
			        filterproduct.addObject(trigger::TriggerL1ETM, myref); 
			      break;
			    case l1t::EtSum::kMissingHt: 
            if(!listHTM.empty())
			        filterproduct.addObject(trigger::TriggerL1HTM, myref); 
			      break;
			    case l1t::EtSum::kMissingEtHF: 
            if(!listETMHF.empty())
			        filterproduct.addObject(trigger::TriggerL1ETMHF, myref); 
			      break;
			    case l1t::EtSum::kCentrality: 
            if(!listCentrality.empty())
			        filterproduct.addObject(trigger::TriggerL1Centrality, myref); 
			      break;
			    case l1t::EtSum::kMinBiasHFP0: 
            if(!listMinBiasHFP0.empty())
			        filterproduct.addObject(trigger::TriggerL1MinBiasHFP0, myref); 
			      break;
			    case l1t::EtSum::kMinBiasHFM0: 
            if(!listMinBiasHFM0.empty())
			        filterproduct.addObject(trigger::TriggerL1MinBiasHFM0, myref); 
			      break;
			    case l1t::EtSum::kMinBiasHFP1: 
            if(!listMinBiasHFP1.empty())
			        filterproduct.addObject(trigger::TriggerL1MinBiasHFP1, myref); 
			      break;
			    case l1t::EtSum::kMinBiasHFM1: 
            if(!listMinBiasHFM1.empty())
			        filterproduct.addObject(trigger::TriggerL1MinBiasHFM1, myref); 
			      break;
			    case l1t::EtSum::kTotalEtEm: 
            if(!listTotalEtEm.empty())
			        filterproduct.addObject(trigger::TriggerL1TotalEtEm, myref); 
			      break;
			    case l1t::EtSum::kTowerCount: 
            if(!listTowerCount.empty())
			        filterproduct.addObject(trigger::TriggerL1TowerCount, myref); 
			      break;
			    case l1t::EtSum::kAsymEt: 
            if(!listAsymEt.empty())
			        filterproduct.addObject(trigger::TriggerL1AsymEt, myref); 
			      break;
			    case l1t::EtSum::kAsymHt: 
            if(!listAsymHt.empty())
			        filterproduct.addObject(trigger::TriggerL1AsymHt, myref); 
			      break;
			    case l1t::EtSum::kAsymEtHF: 
            if(!listAsymEtHF.empty())
			        filterproduct.addObject(trigger::TriggerL1AsymEtHF, myref); 
			      break;
			    case l1t::EtSum::kAsymHtHF: 
            if(!listAsymHtHF.empty())
			        filterproduct.addObject(trigger::TriggerL1AsymHtHF, myref); 
			      break;
			    default:
			      LogTrace("HLTL1TSeed") << "  L1EtSum seed of currently unsuported HLT TriggerType. l1t::EtSum type:      " << iter->getType() << "\n";

			  } // end switch

			} // end for

		} // end else


    // TODO FIXME uncomment if block when JetCounts implemented

    //    // jet counts
    //    if (!listJetCounts.empty()) {
    //        edm::Handle<l1extra::L1JetCounts> l1JetCounts;
    //        iEvent.getByToken(m_l1CollectionsToken.label(), l1JetCounts);
    //
    //        for (std::list<int>::const_iterator itObj = listJetCounts.begin();
    //                itObj != listJetCounts.end(); ++itObj) {
    //
    //            filterproduct.addObject(trigger::TriggerL1JetCounts,l1extra::L1JetCountsRefProd(l1JetCounts));
    //                  // FIXME: RefProd!
    //
    //        }
    //
    //    }


    LogTrace("HLTL1TSeed")
    << "\nHLTL1Seed:seedsL1TriggerObjectMaps returning " << seedsResult << endl << endl;

    return seedsResult;

}

// register as framework plugin
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTL1TSeed);
