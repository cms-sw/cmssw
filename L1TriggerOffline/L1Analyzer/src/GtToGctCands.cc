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
// $Id: GtToGctCands.cc,v 1.4 2009/04/02 19:33:44 tapper Exp $
//
//

#include "L1TriggerOffline/L1Analyzer/interface/GtToGctCands.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

GtToGctCands::GtToGctCands(const edm::ParameterSet& iConfig) :
  m_GTInputTag(iConfig.getParameter<edm::InputTag>("inputLabel"))
{
  // For now I am making one electron collection and one jet collection with all electrons and jets from all 3 BXs.
  // This is the easiest format to analyse for CRAFT data.
  // In the future I should make different collections and treat the mutiple BXs properly, and add energy sums.

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
  // For now only make non-iso electrons and tau jet collections to be fed into L1Extra
  for (int ibx=-1; ibx<=1; ibx++) {
    const L1GtPsbWord psb = gtrr->gtPsbWord(0xbb0d, ibx);
    
     std::vector<int> psbel;
     psbel.push_back(psb.aData(4));
     psbel.push_back(psb.aData(5));
     psbel.push_back(psb.bData(4));
     psbel.push_back(psb.bData(5));
     for(std::vector<int>::const_iterator ipsbel=psbel.begin(); ipsbel!=psbel.end(); ipsbel++) {
       nonIsoEm->push_back(L1GctEmCand((*ipsbel),false)); // set all to non-isolated
     }
    
     std::vector<int> psbjet;
     psbjet.push_back(psb.aData(2));
     psbjet.push_back(psb.aData(3));
     psbjet.push_back(psb.bData(2));
     psbjet.push_back(psb.bData(3));
     for(std::vector<int>::const_iterator ipsbjet=psbjet.begin(); ipsbjet!=psbjet.end(); ipsbjet++) {
       tauJet->push_back(L1GctJetCand((*ipsbjet),true,false)); // set all to tau jets
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
