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
// $Id$
//
//

#include "L1TriggerOffline/L1Analyzer/interface/GtToGctCands.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"


GtToGctCands::GtToGctCands(const edm::ParameterSet& iConfig) :
  m_GTInputTag(iConfig.getParameter<edm::InputTag>("inputLabel"))
{
  // For now I am making one electron collection and one jet collection with all electrons and jets from all 3 BXs.
  // This is the easiest format to analyse for CRAFT data.
  // In the future I should make different collections and treat the mutiple BXs properly, and add energy sums.
  produces<L1GctEmCandCollection>("GT");
  produces<L1GctJetCandCollection>("GT");
}

GtToGctCands::~GtToGctCands(){}

void GtToGctCands::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // create em and jet collections
  std::auto_ptr<L1GctEmCandCollection> allElectrons (new L1GctEmCandCollection());
  std::auto_ptr<L1GctJetCandCollection> allJets (new L1GctJetCandCollection());

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
       allElectrons->push_back(L1GctEmCand((*ipsbel)&0x3f,((*ipsbel)>>10)&0x1f,(((*ipsbel)>>6)&7) * ( ((*ipsbel>>9)&1) ? -1 : 1 ),0));
     }
    
     std::vector<int> psbjet;
     psbjet.push_back(psb.aData(2));
     psbjet.push_back(psb.aData(3));
     psbjet.push_back(psb.bData(2));
     psbjet.push_back(psb.bData(3));
     for(std::vector<int>::const_iterator ipsbjet=psbjet.begin(); ipsbjet!=psbjet.end(); ipsbjet++) {
       allJets->push_back(L1GctJetCand((*ipsbjet)&0x3f,((*ipsbjet)>>10)&0x1f,(((*ipsbjet)>>6)&7) * ( ((*ipsbjet>>9)&1) ? -1 : 1 ),1,0));
     }
  }

  // Put the new collections into the event
  iEvent.put(allElectrons);
  iEvent.put(allJets);
}

//define this as a plug-in
DEFINE_FWK_MODULE(GtToGctCands);
