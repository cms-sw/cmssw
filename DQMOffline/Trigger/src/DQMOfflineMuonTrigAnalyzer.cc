// -*- C++ -*-
//
// Package:     OfflineDQMMuonTrigAnalyzer
// Class:       OfflineDQMMuonTrigAnalyzer
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

#include "DQMOffline/Trigger/interface/HLTMuonGenericRate.h"
#include "DQMOffline/Trigger/interface/HLTMuonOverlap.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "TFile.h"
#include "TDirectory.h"

class OfflineDQMMuonTrigAnalyzer : public edm::EDAnalyzer {

public:
  explicit OfflineDQMMuonTrigAnalyzer(const edm::ParameterSet&);
  ~OfflineDQMMuonTrigAnalyzer();

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



OfflineDQMMuonTrigAnalyzer::OfflineDQMMuonTrigAnalyzer(const ParameterSet& pset)
{

  LogTrace ("HLTMuonVal") << "\n\n Inside MuonTriggerRate Constructor\n\n";
  
  vector<string> triggerNames = pset.getParameter< vector<string> >
                                ("TriggerNames");
  string theHltProcessName = pset.getParameter<string>("HltProcessName");

  HLTConfigProvider hltConfig;
  hltConfig.init(theHltProcessName);
  vector<string> validTriggerNames = hltConfig.triggerNames();

  for( size_t i = 0; i < triggerNames.size(); i++) {
    bool isValidTriggerName = false;
    for ( size_t j = 0; j < validTriggerNames.size(); j++ )
      if ( triggerNames[i] == validTriggerNames[j] ) isValidTriggerName = true;
    if ( !isValidTriggerName ) {}   
    else {
      vector<string> moduleNames = hltConfig.moduleLabels( triggerNames[i] );
      HLTMuonGenericRate *analyzer;
      analyzer = new HLTMuonGenericRate( pset, triggerNames[i], moduleNames );
      theTriggerAnalyzers.push_back( analyzer );
    }
  }
  theOverlapAnalyzer = new HLTMuonOverlap( pset );    

  theNumberOfTriggers = theTriggerAnalyzers.size();  
}


OfflineDQMMuonTrigAnalyzer::~OfflineDQMMuonTrigAnalyzer()
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
OfflineDQMMuonTrigAnalyzer::analyze(const Event& iEvent, const EventSetup& iSetup)
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
OfflineDQMMuonTrigAnalyzer::beginJob(const EventSetup&)
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
OfflineDQMMuonTrigAnalyzer::endJob() {
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
DEFINE_FWK_MODULE(OfflineDQMMuonTrigAnalyzer);
