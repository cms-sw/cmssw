// -*- C++ -*-
//
// Package:    MuonTriggerRateTimeAnalyzer
// Class:      MuonTriggerRateTimeAnalyzer
// 
/**\class MuonTriggerRateTimeAnalyzer MuonTriggerRateTimeAnalyzer.cc HLTriggerOffline/Muon/src/MuonTriggerRateTimeAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Muriel Vander Donckt
//         Created:  Tue Jul 24 12:17:12 CEST 2007
// $Id: MuonTriggerRateTimeAnalyzer.cc,v 1.11 2009/01/06 19:22:27 klukas Exp $
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "HLTriggerOffline/Muon/interface/HLTMuonGenericRate.h"
#include "HLTriggerOffline/Muon/interface/HLTMuonOverlap.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "TFile.h"
#include "TDirectory.h"

class MuonTriggerRateTimeAnalyzer : public edm::EDAnalyzer {

public:
  explicit MuonTriggerRateTimeAnalyzer(const edm::ParameterSet&);
  ~MuonTriggerRateTimeAnalyzer();

private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  int theNumberOfTriggers;
  std::vector<HLTMuonGenericRate*> theTriggerAnalyzers;
  HLTMuonOverlap *theOverlapAnalyzer;

};

using namespace std;
using namespace edm;



MuonTriggerRateTimeAnalyzer::MuonTriggerRateTimeAnalyzer(const ParameterSet& pset)
{

  string theHltProcessName = pset.getParameter<string>("HltProcessName");

  HLTConfigProvider hltConfig;
  hltConfig.init(theHltProcessName);
  vector<string> triggerNames = hltConfig.triggerNames();
  vector<string> muonTriggerNames;

  for( size_t i = 0; i < triggerNames.size(); i++) {
    TString triggerName = triggerNames[i];
    if (triggerName.Contains("Mu")) { 
      TString triggerNameWithoutPrefix = triggerName(4,triggerName.Length());
      // Do not accept crossed triggers 
      if (!triggerNameWithoutPrefix.Contains("_")) 
	muonTriggerNames.push_back(triggerNames[i]);
    }
  }

  for( size_t i = 0; i < muonTriggerNames.size(); i++) {
    vector<string> moduleNames = hltConfig.moduleLabels( muonTriggerNames[i] );
    HLTMuonGenericRate *analyzer;
    analyzer = new HLTMuonGenericRate( pset, muonTriggerNames[i], moduleNames );
    theTriggerAnalyzers.push_back( analyzer );
  }

  theOverlapAnalyzer = new HLTMuonOverlap( pset );    

  theNumberOfTriggers = theTriggerAnalyzers.size();  
}


MuonTriggerRateTimeAnalyzer::~MuonTriggerRateTimeAnalyzer()
{
  vector<HLTMuonGenericRate *>::iterator thisAnalyzer;
  for ( thisAnalyzer  = theTriggerAnalyzers.begin(); 
        thisAnalyzer != theTriggerAnalyzers.end(); 
	++thisAnalyzer )
  {
    delete *thisAnalyzer;
  } 
  theTriggerAnalyzers.clear();
  delete theOverlapAnalyzer;
}


//
// member functions
//

void
MuonTriggerRateTimeAnalyzer::analyze(const Event& iEvent, const EventSetup& iSetup)
{
  vector<HLTMuonGenericRate *>::iterator thisAnalyzer;
  for ( thisAnalyzer  = theTriggerAnalyzers.begin(); 
	thisAnalyzer != theTriggerAnalyzers.end(); ++thisAnalyzer )
    {
      (*thisAnalyzer)->analyze(iEvent);
    } 
  theOverlapAnalyzer ->analyze(iEvent);
}



void 
MuonTriggerRateTimeAnalyzer::beginJob(const EventSetup&)
{
  vector<HLTMuonGenericRate *>::iterator thisAnalyzer;
  for ( thisAnalyzer  = theTriggerAnalyzers.begin(); 
        thisAnalyzer != theTriggerAnalyzers.end(); 
	++thisAnalyzer )
    {
      (*thisAnalyzer)->begin();
    } 
  theOverlapAnalyzer ->begin();
}



void 
MuonTriggerRateTimeAnalyzer::endJob() {
  vector<HLTMuonGenericRate *>::iterator thisAnalyzer;
  for ( thisAnalyzer  = theTriggerAnalyzers.begin(); 
        thisAnalyzer != theTriggerAnalyzers.end(); 
	++thisAnalyzer )
    {
      (*thisAnalyzer)->finish();
    }
  theOverlapAnalyzer ->finish();
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonTriggerRateTimeAnalyzer);
