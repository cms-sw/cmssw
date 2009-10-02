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
// Jason Slaunwhite, based on code from Jeff Klukas
// $Id: DQMOfflineMuonTrigAnalyzer.cc,v 1.10 2009/08/25 10:03:15 slaunwhj Exp $
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

#include "DQMOffline/Trigger/interface/HLTMuonMatchAndPlot.h"
#include "DQMOffline/Trigger/interface/HLTMuonOverlap.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
//#include "PhysicsTools/Utilities/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "TFile.h"
#include "TDirectory.h"





class OfflineDQMMuonTrigAnalyzer : public edm::EDAnalyzer {

public:
  explicit OfflineDQMMuonTrigAnalyzer(const edm::ParameterSet&);
  ~OfflineDQMMuonTrigAnalyzer();

private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endRun (const edm::Run& r, const edm::EventSetup& c);
  virtual void endJob() ;

  int theNumberOfTriggers;
  std::vector<HLTMuonMatchAndPlot*> theTriggerAnalyzers;
  //HLTMuonOverlap *theOverlapAnalyzer;

};

using namespace std;
using namespace edm;
using reco::Muon;


OfflineDQMMuonTrigAnalyzer::OfflineDQMMuonTrigAnalyzer(const ParameterSet& pset)
{

  LogTrace ("HLTMuonVal") << "\n\n Inside MuonTriggerRate Constructor\n\n";
  
  vector<string> triggerNames = pset.getParameter< vector<string> >
                                ("TriggerNames");
  string theHltProcessName = pset.getParameter<string>("HltProcessName");

  //string defRecoLabel = pset.getUntrackedParameter<string>("RecoLabel","");
  //string highPtTracksLabel =  pset.getParameter <string> ("highPtTrackCollection");

  //vector<string> recoCollectionNames = pset.getParameter < vector<string> > ("allCollectionNames");

  
  // make analyzers for each collection. Push the collections into a vector
  //vector <string> recoCollectionNames;
  //if (defRecoLabel != "") recoCollectionNames.push_back(defRecoLabel);
  //if (highPtTracksLabel != "") recoCollectionNames.push_back(highPtTracksLabel);

  vector<edm::ParameterSet> customCollection = pset.getParameter<vector<edm::ParameterSet> > ("customCollection");

  vector<edm::ParameterSet>::iterator iPSet;

  LogTrace ("HLTMuonVal") << "customCollection is a vector of size = " << customCollection.size() << std::endl
                          << "looping over entries" << std::endl;

    
  DQMStore * dbe_ = 0;
  if ( pset.getUntrackedParameter<bool>("DQMStore", false) ) {
    dbe_ = Service<DQMStore>().operator->();
    dbe_->setVerbose(0);    
  }


  vector < MuonSelectionStruct > customSelectors;
  vector < string > customNames;
  // Print out information about each pset
  for ( iPSet = customCollection.begin();
        iPSet != customCollection.end();
        iPSet++) {
    string customCuts  = iPSet->getUntrackedParameter<string> ("recoCuts");
    string customName = iPSet->getUntrackedParameter<string> ("collectionName");
    string hltCuts   = iPSet->getUntrackedParameter<string> ("hltCuts");
    string targetTrackCollection = iPSet->getUntrackedParameter<string> ("trackCollection");
    double  customD0Cut = iPSet->getUntrackedParameter<double> ("d0cut");
    double customZ0Cut = iPSet->getUntrackedParameter<double> ("z0cut");

    vector<string> requiredTriggers = iPSet->getUntrackedParameter< vector<string> > ("requiredTriggers");
    
    LogTrace("HLTMuonVal") << "customTargetCollection = " << customName  << std::endl
                           << "customCuts = " << customCuts << std::endl
                           << "targetTrackCollection = " << targetTrackCollection << std::endl
                           << "d0 cut = " << customD0Cut << std::endl
                           << "z0 cut = " << customZ0Cut << std:: endl ;

    if (dbe_) {

      string description = customName + ", reco cuts = " + customCuts
        + ", hlt cuts = " + hltCuts + ", trackCollection = " + targetTrackCollection
        + ", required triggers, ";

      // add the required triggers
      for (vector <string>::const_iterator trigit = requiredTriggers.begin();
           trigit != requiredTriggers.end();
           trigit++){
        description  += (*trigit) + ", ";
      }

      // Add the other cuts
      ostringstream ossd0, ossz0, osschi2, osshits;

      
      ossd0 << customD0Cut;
      ossz0 << customZ0Cut;
      //osschi2 << customChi2Cut;
      //osshits << customNHitsCut;

      description += "|d0| < "  + ossd0.str() + ", |z0| < " + ossz0.str();
        

      LogTrace ("HLTMuonVal") << "Storing description = " << description << endl;

      dbe_->setCurrentFolder("HLT/Muon/Distributions/");

      dbe_->bookString (customName, description);            

    }


    StringCutObjectSelector<Muon> tempRecoSelector(customCuts);
    StringCutObjectSelector<TriggerObject> tempHltSelector(hltCuts);
    
    // create a custom selector
    MuonSelectionStruct tempStruct(tempRecoSelector, tempHltSelector,
                                   customName, customD0Cut, customZ0Cut,
                                   targetTrackCollection, requiredTriggers);

    
    customNames.push_back ( customName);
    customSelectors.push_back(tempStruct);
  }

  
  
  LogTrace ("HLTMuonVal") << "Initializing HLTConfigProvider with HLT process name: " << theHltProcessName << endl;
  HLTConfigProvider hltConfig;
  hltConfig.init(theHltProcessName);
  vector<string> validTriggerNames = hltConfig.triggerNames();

  if (validTriggerNames.size() < 1) {
    LogInfo ("HLTMuonVal") << endl << endl << endl
                           << "---> WARNING: The HLT Config Provider gave you an empty list of valid trigger names" << endl
                           << "Could be a problem with the HLT Process Name (you provided  " << theHltProcessName <<")" << endl
                           << "W/o valid triggers we can't produce plots, exiting..."
                           << endl << endl << endl;
    
    // don't return... you'll automatically skip the rest
    //return;
  }

  vector<string>::const_iterator iDumpName;
  unsigned int numTriggers = 0;
  for (iDumpName = validTriggerNames.begin();
       iDumpName != validTriggerNames.end();
       iDumpName++) {

    LogTrace ("HLTMuonVal") << "Trigger " << numTriggers
                            << " is called " << (*iDumpName)
                            << endl;
    numTriggers++;
  }


  vector<MuonSelectionStruct>::iterator iMuonSelector;
  vector<string>::iterator iName = customNames.begin();
  for ( iMuonSelector = customSelectors.begin();
        iMuonSelector != customSelectors.end();
        iMuonSelector++) {
  
    for( size_t i = 0; i < triggerNames.size(); i++) {
      bool isValidTriggerName = false;
      for ( size_t j = 0; j < validTriggerNames.size(); j++ )
        if ( triggerNames[i] == validTriggerNames[j] ) isValidTriggerName = true;
      if ( !isValidTriggerName ) {}   
      else {
        vector<string> moduleNames = hltConfig.moduleLabels( triggerNames[i] );
        HLTMuonMatchAndPlot *analyzer;
        analyzer = new HLTMuonMatchAndPlot( pset, triggerNames[i], moduleNames, (*iMuonSelector), (*iName), validTriggerNames );
        theTriggerAnalyzers.push_back( analyzer );
      }
    }
    iName++;
  }
  //theOverlapAnalyzer = new HLTMuonOverlap( pset );    

  theNumberOfTriggers = theTriggerAnalyzers.size();

  LogTrace ("HLTMuonVal") << "You have created " << theNumberOfTriggers
                          << " trigger analyzers" << endl;
}


OfflineDQMMuonTrigAnalyzer::~OfflineDQMMuonTrigAnalyzer()
{

  LogTrace ("HLTMuonVal")
    << "Inside OfflineDQMMuonTrigAnalyzer destructor" << endl;
  
  vector<HLTMuonMatchAndPlot *>::iterator thisAnalyzer;
  for ( thisAnalyzer  = theTriggerAnalyzers.begin(); 
        thisAnalyzer != theTriggerAnalyzers.end(); 
	++thisAnalyzer )
  {
    delete *thisAnalyzer;
  } 
  theTriggerAnalyzers.clear();
  //delete theOverlapAnalyzer;
}


//
// member functions
//

void
OfflineDQMMuonTrigAnalyzer::analyze(const Event& iEvent, const EventSetup& iSetup)
{
  LogTrace ("HLTMuonVal")
    << "Inside OfflineDQMMuonTrigAnalyzer analyze" << endl;
  
  vector<HLTMuonMatchAndPlot *>::iterator thisAnalyzer;
  for ( thisAnalyzer  = theTriggerAnalyzers.begin(); 
	thisAnalyzer != theTriggerAnalyzers.end(); ++thisAnalyzer )
    {
      (*thisAnalyzer)->analyze(iEvent);
    } 
  //theOverlapAnalyzer ->analyze(iEvent);
}



void 
OfflineDQMMuonTrigAnalyzer::beginJob()
{

  LogTrace ("HLTMuonVal")
    << "Inside OfflineDQMMuonTrigAnalyzer beginJob" << endl;

  LogTrace ("HLTMuonVal")
    << "Looking at a vector of analyzers, with size " << theTriggerAnalyzers.size() << endl;


  vector<HLTMuonMatchAndPlot *>::iterator thisAnalyzer;
  int nTotalAnalyzers = 0;
  for ( thisAnalyzer  = theTriggerAnalyzers.begin(); 
        thisAnalyzer != theTriggerAnalyzers.end(); 
	++thisAnalyzer )
    {

      LogTrace ("HLTMuonVal")
        << "Calling begin for analyzer " << nTotalAnalyzers;
       
      (*thisAnalyzer)->begin();

      
      nTotalAnalyzers++;
    }

  LogTrace ("HLTMuonVal")
    << "OfflineDQMMuonTrigAnalyzer: Calling being for overlap analyzer" << endl;
  
  //theOverlapAnalyzer ->begin();
}

void 
OfflineDQMMuonTrigAnalyzer::endRun ( const edm::Run& theRun, const edm::EventSetup& theEventSetup) {

  LogTrace ("HLTMuonVal")
    << "Inside OfflineDQMMuonTrigAnalyzer endRun()" << endl;
  
  
  vector<HLTMuonMatchAndPlot *>::iterator thisAnalyzer;
  for ( thisAnalyzer  = theTriggerAnalyzers.begin(); 
        thisAnalyzer != theTriggerAnalyzers.end(); 
	++thisAnalyzer )
    {
      (*thisAnalyzer)->endRun(theRun, theEventSetup);
    }

}



void 
OfflineDQMMuonTrigAnalyzer::endJob() {

  LogTrace ("HLTMuonVal")
    << "Inside OfflineDQMMuonTrigAnalyzer endJob()" << endl;
  
  
  vector<HLTMuonMatchAndPlot *>::iterator thisAnalyzer;
  for ( thisAnalyzer  = theTriggerAnalyzers.begin(); 
        thisAnalyzer != theTriggerAnalyzers.end(); 
	++thisAnalyzer )
    {
      (*thisAnalyzer)->finish();
    }
  //theOverlapAnalyzer ->finish();
}

//define this as a plug-in
DEFINE_FWK_MODULE(OfflineDQMMuonTrigAnalyzer);
