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
// $Id: MuonTriggerRateTimeAnalyzer.cc,v 1.10 2008/11/10 20:34:53 klukas Exp $
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
