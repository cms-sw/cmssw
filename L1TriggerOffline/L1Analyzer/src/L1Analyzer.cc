// -*- C++ -*-
//
// Package:    L1Analyzer
// Class:      L1Analyzer
// 
/**\class L1Analyzer L1Analyzer.cc L1TriggerOffline/L1Analyzer/src/L1Analyzer.cc

 Description: Analyze the GCT output.

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Alex Tapper
//         Created:  Thu Nov 30 20:57:38 CET 2006
// $Id: L1Analyzer.cc,v 1.2 2007/07/08 08:14:05 elmer Exp $
//
//


// system include files
#include <memory>

// user include files
#include "L1TriggerOffline/L1Analyzer/interface/L1Analyzer.h"

// Data formats 
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"

using namespace reco;
using namespace edm;
using namespace std;

//
// constructors and destructor
//

L1Analyzer::L1Analyzer(const edm::ParameterSet& iConfig):
  m_candidateSource(iConfig.getUntrackedParameter<edm::InputTag>("CandidateSource")),
  m_referenceSource(iConfig.getUntrackedParameter<edm::InputTag>("ReferenceSource")),
  m_resMatchMapSource(iConfig.getUntrackedParameter<edm::InputTag>("ResMatchMapSource")),
  m_effMatchMapSource(iConfig.getUntrackedParameter<edm::InputTag>("EffMatchMapSource"))
{
  m_l1UnMatched  = new SimpleHistograms("L1Candidates",iConfig);
  m_refUnMatched = new SimpleHistograms("RefCandidates",iConfig); 
  m_l1Matched    = new SimpleHistograms("L1MatchedCandidates",iConfig);
  m_refMatched   = new SimpleHistograms("RefMatchedCandidates",iConfig); 
  m_resolution   = new ResolutionHistograms("Resolutions",iConfig); 
  m_efficiency   = new EfficiencyHistograms("Efficiencies",iConfig); 
}

L1Analyzer::~L1Analyzer()
{
  delete m_l1UnMatched;
  delete m_refUnMatched;
  delete m_l1Matched;
  delete m_refMatched;
  delete m_resolution;
  delete m_efficiency;
}

void L1Analyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  // Get the L1 candidates from the event
  Handle<CandidateCollection> Cands;
  iEvent.getByLabel(m_candidateSource,Cands);

  // Get the resolution matching map from the event
  Handle<CandMatchMap> ResMatchMap;
  iEvent.getByLabel(m_resMatchMapSource,ResMatchMap);

  // Loop over the L1 candidates looking for a match
  for (unsigned i=0; i<Cands->size(); i++){
    CandidateRef CandRef(Cands,i);

    // Fill unmatched histogram
    m_l1UnMatched->Fill(CandRef);

    // Loop over match map
    CandMatchMap::const_iterator f = ResMatchMap->find(CandRef);
    if (f!=ResMatchMap->end()){
      const CandidateRef &CandMatch = f->val;

      // Fill the histograms
      m_l1Matched->Fill(CandRef);
      m_refMatched->Fill(CandMatch);
      m_resolution->Fill(CandRef,CandMatch);
    }
  }   

  // Get the reference collection (either MC truth or RECO) from the event
  Handle<CandidateCollection> Refs;
  iEvent.getByLabel(m_referenceSource,Refs);

  // Get the efficiency matching map from the event
  Handle<CandMatchMap> EffMatchMap;
  iEvent.getByLabel(m_effMatchMapSource,EffMatchMap);

  // Loop over the reference collection looking for a match
  for (unsigned i=0; i<Refs->size(); i++){
    CandidateRef CandRef(Refs,i);

    // Fill the unmatched histograms
    m_refUnMatched->Fill(CandRef);

    // Fill the efficiency histograms
    m_efficiency->FillReference(CandRef);

    // See if this reference candidate was matched 
    CandMatchMap::const_iterator f = EffMatchMap->find(CandRef);
    if (f!=EffMatchMap->end()){
      // Fill the efficiency histograms
      m_efficiency->FillL1(CandRef);
    }
  }   

}







