// -*- C++ -*-
//
// Package:    GtToGctCands
// Class:      GtToGctCands
// 
/**\class GtToGctCands GtToGctCands.cc L1TriggerOffline/L1Analyzer/src/GtToGctCands.cc

 Description: Convert GT candidates (electrons and jets) to GCT format

*/
//
// Original Author:  Alex Tapper
//         Created:  Mon Mar 30 17:31:03 CEST 2009
// $Id: GtToGctCands.cc,v 1.5 2009/04/05 16:19:20 tapper Exp $
//
//

#include "L1TriggerOffline/L1Analyzer/interface/GtToGctCands.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

GtToGctCands::GtToGctCands(const edm::ParameterSet& iConfig) :
  m_GTInputTag(iConfig.getParameter<edm::InputTag>("inputLabel"))
{
  // For now I am making one collection from all 3 BXs.
  // This is the easiest format to analyse for CRAFT data.
  // In the future I should treat the mutiple BXs properly, and add energy sums.

  // list of products
  produces<L1GctEmCandCollection>("isoEm");
  produces<L1GctEmCandCollection>("nonIsoEm");
  produces<L1GctJetCandCollection>("cenJets");
  produces<L1GctJetCandCollection>("forJets");
  produces<L1GctJetCandCollection>("tauJets");
  produces<L1GctEtTotalCollection>();
  produces<L1GctEtHadCollection>();
  produces<L1GctEtMissCollection>();
  //  produces<L1GctHtMissCollection>();
  produces<L1GctHFBitCountsCollection>();
  produces<L1GctHFRingEtSumsCollection>();
}

GtToGctCands::~GtToGctCands(){}

void GtToGctCands::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // create the em and jet collections
  std::auto_ptr<L1GctEmCandCollection>  isoEm    (new L1GctEmCandCollection);
  std::auto_ptr<L1GctEmCandCollection>  nonIsoEm (new L1GctEmCandCollection);
  std::auto_ptr<L1GctJetCandCollection> cenJet   (new L1GctJetCandCollection);
  std::auto_ptr<L1GctJetCandCollection> forJet   (new L1GctJetCandCollection);
  std::auto_ptr<L1GctJetCandCollection> tauJet   (new L1GctJetCandCollection);

  // create the energy sum digis
  std::auto_ptr<L1GctEtTotalCollection> etTot  (new L1GctEtTotalCollection);
  std::auto_ptr<L1GctEtHadCollection>   etHad  (new L1GctEtHadCollection);
  std::auto_ptr<L1GctEtMissCollection>  etMiss (new L1GctEtMissCollection);
//  std::auto_ptr<L1GctHtMissCollection>  htMiss (new L1GctHtMissCollection));

  // create the Hf sums digis
  std::auto_ptr<L1GctHFBitCountsCollection>  hfBitCount  (new L1GctHFBitCountsCollection);
  std::auto_ptr<L1GctHFRingEtSumsCollection> hfRingEtSum (new L1GctHFRingEtSumsCollection);

  // Get GT data
  edm::Handle<L1GlobalTriggerReadoutRecord> gtrr_handle;
  iEvent.getByLabel(m_GTInputTag,gtrr_handle);
  L1GlobalTriggerReadoutRecord const* gtrr = gtrr_handle.product();

  // Loop over 3BXs (shouldn't be hard coded really) and get GT cands
  for (int ibx=-1; ibx<=1; ibx++) {
    const L1GtPsbWord psb1 = gtrr->gtPsbWord(0xbb0d, ibx); 
    const L1GtPsbWord psb2 = gtrr->gtPsbWord(0xbb0e, ibx); 
    
    // Isolated electrons
    std::vector<int> psbisoel;
    psbisoel.push_back(psb1.aData(6));
    psbisoel.push_back(psb1.aData(7));
    psbisoel.push_back(psb1.bData(6));
    psbisoel.push_back(psb1.bData(7));
    for(std::vector<int>::const_iterator ipsbisoel=psbisoel.begin(); ipsbisoel!=psbisoel.end(); ipsbisoel++) {
      isoEm->push_back(L1GctEmCand((*ipsbisoel),true)); 
    }

    // Non-isolated electrons
    std::vector<int> psbel;
    psbel.push_back(psb1.aData(4));
    psbel.push_back(psb1.aData(5));
    psbel.push_back(psb1.bData(4));
    psbel.push_back(psb1.bData(5));
    for(std::vector<int>::const_iterator ipsbel=psbel.begin(); ipsbel!=psbel.end(); ipsbel++) {
      nonIsoEm->push_back(L1GctEmCand((*ipsbel),false)); 
    }

    // Central jets
    std::vector<int> psbjet;
    psbjet.push_back(psb1.aData(2));
    psbjet.push_back(psb1.aData(3));
    psbjet.push_back(psb1.bData(2));
    psbjet.push_back(psb1.bData(3));
    for(std::vector<int>::const_iterator ipsbjet=psbjet.begin(); ipsbjet!=psbjet.end(); ipsbjet++) {
      cenJet->push_back(L1GctJetCand((*ipsbjet),false,false));
    }

    // Forward jets
    std::vector<int> psbfjet;
    psbfjet.push_back(psb1.aData(6));
    psbfjet.push_back(psb1.aData(7));
    psbfjet.push_back(psb1.bData(6));
    psbfjet.push_back(psb1.bData(7));
    for(std::vector<int>::const_iterator ipsbfjet=psbfjet.begin(); ipsbfjet!=psbfjet.end(); ipsbfjet++) {
      forJet->push_back(L1GctJetCand((*ipsbfjet),false,true));
    }

    // Tau jets
    std::vector<int> psbtjet;
    psbtjet.push_back(psb2.aData(6));
    psbtjet.push_back(psb2.aData(7));
    psbtjet.push_back(psb2.bData(6));
    psbtjet.push_back(psb2.bData(7));
    for(std::vector<int>::const_iterator ipsbtjet=psbtjet.begin(); ipsbtjet!=psbtjet.end(); ipsbtjet++) {
      tauJet->push_back(L1GctJetCand((*ipsbtjet),true,false));
    }

  }
  
  // put the collections into the event
  iEvent.put(isoEm,"isoEm");
  iEvent.put(nonIsoEm,"nonIsoEm");
  iEvent.put(cenJet,"cenJets");
  iEvent.put(forJet,"forJets");
  iEvent.put(tauJet,"tauJets");
  iEvent.put(etTot);
  iEvent.put(etHad);
  iEvent.put(etMiss);
//  iEvent.put(htMiss);
  iEvent.put(hfBitCount);
  iEvent.put(hfRingEtSum);
}
