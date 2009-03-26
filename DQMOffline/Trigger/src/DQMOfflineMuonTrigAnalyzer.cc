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
// $Id: DQMOfflineMuonTrigAnalyzer.cc,v 1.1 2009/02/27 13:12:09 slaunwhj Exp $
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

  //string defRecoLabel = pset.getUntrackedParameter<string>("RecoLabel","");
  //string highPtTracksLabel =  pset.getParameter <string> ("highPtTrackCollection");

  vector<string> recoCollectionNames = pset.getParameter < vector<string> > ("allCollectionNames");

  
  // make analyzers for each collection. Push the collections into a vector
  //vector <string> recoCollectionNames;
  //if (defRecoLabel != "") recoCollectionNames.push_back(defRecoLabel);
  //if (highPtTracksLabel != "") recoCollectionNames.push_back(highPtTracksLabel);

  
  
  HLTConfigProvider hltConfig;
  hltConfig.init(theHltProcessName);
  vector<string> validTriggerNames = hltConfig.triggerNames();

  vector<string>::const_iterator iRecCollection;

  for ( iRecCollection = recoCollectionNames.begin();
        iRecCollection != recoCollectionNames.end();
        iRecCollection++) {
  
    for( size_t i = 0; i < triggerNames.size(); i++) {
      bool isValidTriggerName = false;
      for ( size_t j = 0; j < validTriggerNames.size(); j++ )
        if ( triggerNames[i] == validTriggerNames[j] ) isValidTriggerName = true;
      if ( !isValidTriggerName ) {}   
      else {
        vector<string> moduleNames = hltConfig.moduleLabels( triggerNames[i] );
        HLTMuonGenericRate *analyzer;
        analyzer = new HLTMuonGenericRate( pset, triggerNames[i], moduleNames, (*iRecCollection) );
        theTriggerAnalyzers.push_back( analyzer );
      }
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
