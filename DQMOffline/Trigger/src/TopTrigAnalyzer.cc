// -*- C++ -*-
//
// Package:     TopTrigAnalyzer
// Class:       TopTrigAnalyzer
// 
/**\class MuonTriggerRateTimeAnalyzer MuonTriggerRateTimeAnalyzer.cc HLTriggerOffline/Muon/src/MuonTriggerRateTimeAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Muriel Vander Donckt
//         Created:  Tue Jul 24 12:17:12 CEST 2007
// $Id: TopTrigAnalyzer.cc,v 1.3 2009/11/03 14:06:20 slaunwhj Exp $
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
#include "DQMOffline/Trigger/interface/HLTTopPlotter.h"
//#include "DQMOffline/Trigger/interface/HLTMuonOverlap.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
//#include "PhysicsTools/Utilities/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include "TFile.h"
#include "TDirectory.h"





class TopTrigAnalyzer : public edm::EDAnalyzer {

public:
  explicit TopTrigAnalyzer(const edm::ParameterSet&);
  ~TopTrigAnalyzer();

private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  virtual void endRun (const edm::Run& r, const edm::EventSetup& c);
  int theNumberOfTriggers;

  std::vector<HLTMuonMatchAndPlot*> theTriggerAnalyzers;
  std::vector<HLTTopPlotter*> theTopPlotters;
  
  //HLTMuonOverlap *theOverlapAnalyzer;

};

using namespace std;
using namespace edm;
using reco::Muon;


TopTrigAnalyzer::TopTrigAnalyzer(const ParameterSet& pset)
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
                          << "looping over entries... and storing descriptions in a root file" << std::endl;

  
  DQMStore * dbe_ = 0;
  if ( pset.getUntrackedParameter<bool>("DQMStore", false) ) {
    dbe_ = Service<DQMStore>().operator->();
    dbe_->setVerbose(0);    
  }

  //////////////////////////////////////////////////
  //
  // Parse the inputs
  //
  ///////////////////////////////////////////////////
  
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

    // the following two parameters are not currently used
    // but maybe in the future
    double customChi2Cut = iPSet->getUntrackedParameter<double> ("chi2cut", 30.0);
    int customNHitsCut = iPSet->getUntrackedParameter<int> ("nHits", 10);

    //
    vector<string> requiredTriggers = iPSet->getUntrackedParameter< vector<string> > ("requiredTriggers");
    
    LogTrace("HLTMuonVal") << "customTargetCollection = " << customName  << std::endl
                           << "customCuts = " << customCuts << std::endl
                           << "targetTrackCollection = " << targetTrackCollection << std::endl
                           << "d0 cut = " << customD0Cut << std::endl
                           << "z0 cut = " << customZ0Cut << std:: endl
                           << "nHits cut = " << customNHitsCut << std::endl
                           << "chi2 cut = " << customChi2Cut <<std::endl;

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
      osschi2 << customChi2Cut;
      osshits << customNHitsCut;

      description += "|d0| < "  + ossd0.str() + ", |z0| < " + ossz0.str()
        + ", chi2 < " + osschi2.str() + ", nHits > " + osshits.str();

      LogTrace ("HLTMuonVal") << "Storing description = " << description << endl;

      dbe_->setCurrentFolder("HLT/Muon/Distributions/");

      dbe_->bookString (customName, description);
      
      

    }

    
    StringCutObjectSelector<Muon> tempRecoSelector(customCuts);
    StringCutObjectSelector<TriggerObject> tempHltSelector(hltCuts);
    
    // create a custom selector
    MuonSelectionStruct tempStruct(tempRecoSelector, tempHltSelector,
                                   customName, customD0Cut, customZ0Cut,
                                   //customChi2Cut, customNHitsCut,
                                   targetTrackCollection, requiredTriggers);

    
    customNames.push_back ( customName);
    customSelectors.push_back(tempStruct);
  }



  
  
  HLTConfigProvider hltConfig;
  bool hltConfigInitSuccess = hltConfig.init(theHltProcessName);
  vector<string> validTriggerNames;

  if (hltConfigInitSuccess)
    validTriggerNames = hltConfig.triggerNames();

  if (validTriggerNames.size() < 1) {
    LogTrace ("HLTMuonVal") << endl << endl << endl
                            << "---> WARNING: The HLT Config Provider gave you an empty list of valid trigger names" << endl
                            << "Could be a problem with the HLT Process Name (you provided  " << theHltProcessName <<")" << endl
                            << "W/o valid triggers we can't produce plots, exiting..."
                            << endl << endl << endl;
    return;
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


  /////////////////////////////////////////////////
  //
  // Create the analyzers
  //
  ////////////////////////////////////////////////

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
        //HLTMuonMatchAndPlot *analyzer;
        //analyzer = new HLTMuonMatchAndPlot ( pset, triggerNames[i], moduleNames, (*iMuonSelector), (*iName), validTriggerNames );
        HLTTopPlotter * tempTop;
        tempTop = new HLTTopPlotter ( pset, triggerNames[i], moduleNames, (*iMuonSelector), (*iName), validTriggerNames);
        //theTriggerAnalyzers.push_back( analyzer );
        theTopPlotters.push_back (tempTop);
      }
    }
    iName++;
  }
  //theOverlapAnalyzer = new HLTMuonOverlap( pset );    

  //theNumberOfTriggers = theTriggerAnalyzers.size();
  theNumberOfTriggers = theTopPlotters.size();  
}


TopTrigAnalyzer::~TopTrigAnalyzer()
{
 //  vector<HLTMuonMatchAndPlot *>::iterator thisAnalyzer;
//   for ( thisAnalyzer  = theTriggerAnalyzers.begin(); 
//         thisAnalyzer != theTriggerAnalyzers.end(); 
// 	++thisAnalyzer )
//   {
//     delete *thisAnalyzer;
//   }

  vector<HLTTopPlotter *>::iterator iTopAna;
  for ( iTopAna  = theTopPlotters.begin(); 
        iTopAna != theTopPlotters.end(); 
	++iTopAna )
  {
    delete *iTopAna;
  }

  theTopPlotters.clear();
  //theTriggerAnalyzers.clear();
  //delete theOverlapAnalyzer;
}


//
// member functions
//

void
TopTrigAnalyzer::analyze(const Event& iEvent, const EventSetup& iSetup)
{
  //vector<HLTMuonMatchAndPlot *>::iterator thisAnalyzer;

  //  unsigned iAna = 0;
//   for ( thisAnalyzer  = theTriggerAnalyzers.begin(); 
// 	thisAnalyzer != theTriggerAnalyzers.end(); ++thisAnalyzer )
//     {
//       (*thisAnalyzer)->analyze(iEvent);
//       theTopPlotters[iAna]->analyze(iEvent);
//       iAna++;
//     }

  vector<HLTTopPlotter *>::iterator iTopAna;
  for ( iTopAna  = theTopPlotters.begin(); 
        iTopAna != theTopPlotters.end(); 
	++iTopAna )
  {
    (*iTopAna)->analyze(iEvent);
  }


  
  //theOverlapAnalyzer ->analyze(iEvent);
}



void 
TopTrigAnalyzer::beginJob()
{
  
  vector<HLTTopPlotter *>::iterator thisAnalyzer;
  //unsigned iAna = 0;
  LogTrace ("HLTMuonVal") << "Inside begin job " << endl
                          << "Looping over analyzers"
                          << endl;
  
  for ( thisAnalyzer  = theTopPlotters.begin(); 
        thisAnalyzer != theTopPlotters.end(); 
	++thisAnalyzer )
    {
     (*thisAnalyzer)->begin();     
    } 
  //theOverlapAnalyzer ->begin();
}

void 
TopTrigAnalyzer::endRun( const edm::Run& theRun, const edm::EventSetup& theEventSetup ) {
  vector<HLTTopPlotter *>::iterator thisAnalyzer;
  //unsigned iAna = 0;

  LogTrace ("HLTMuonVal") << "Inside end job, looping over analyzers"
                          << endl;
  for ( thisAnalyzer  = theTopPlotters.begin(); 
        thisAnalyzer != theTopPlotters.end(); 
	++thisAnalyzer )
    {
      (*thisAnalyzer)->endRun(theRun, theEventSetup);      
    }
  //theOverlapAnalyzer ->finish();
}


void 
TopTrigAnalyzer::endJob() {
  vector<HLTTopPlotter *>::iterator thisAnalyzer;
  //unsigned iAna = 0;

  LogTrace ("HLTMuonVal") << "Inside end job, looping over analyzers"
                          << endl;
  for ( thisAnalyzer  = theTopPlotters.begin(); 
        thisAnalyzer != theTopPlotters.end(); 
	++thisAnalyzer )
    {
      (*thisAnalyzer)->finish();      
    }
  //theOverlapAnalyzer ->finish();
}

//define this as a plug-in
DEFINE_FWK_MODULE(TopTrigAnalyzer);
