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
// $Id: MuonTriggerRateTimeAnalyzer.cc,v 1.8 2008/09/18 20:57:51 klukas Exp $
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
#include "HLTriggerOffline/Muon/interface/HLTMuonTime.h"

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
  HLTMuonTime    *theTimeAnalyzer;

};

using namespace std;
using namespace edm;



MuonTriggerRateTimeAnalyzer::MuonTriggerRateTimeAnalyzer(const ParameterSet& pset)
{

  vector<string> triggerNames = pset.getParameter< vector<string> >
                                ("TriggerNames");
  theNumberOfTriggers     = triggerNames.size();
  for( int i = 0; i < theNumberOfTriggers; i++) {
    HLTMuonGenericRate *analyzer = new HLTMuonGenericRate( pset, triggerNames[i] );
    theTriggerAnalyzers.push_back( analyzer );
  }
  theOverlapAnalyzer = new HLTMuonOverlap( pset );    
  theTimeAnalyzer    = new HLTMuonTime( pset );  
  
}


MuonTriggerRateTimeAnalyzer::~MuonTriggerRateTimeAnalyzer()
{
  using namespace edm;
  vector<HLTMuonGenericRate *>::iterator thisAnalyzer;
  for ( thisAnalyzer  = theTriggerAnalyzers.begin(); 
        thisAnalyzer != theTriggerAnalyzers.end(); 
	++thisAnalyzer )
  {
    delete *thisAnalyzer;
  } 
  theTriggerAnalyzers.clear();
  delete theOverlapAnalyzer;
  delete theTimeAnalyzer;

}


//
// member functions
//

void
MuonTriggerRateTimeAnalyzer::analyze(const Event& iEvent, const EventSetup& iSetup)
{
  using namespace edm;
  vector<HLTMuonGenericRate *>::iterator thisAnalyzer;
  for ( thisAnalyzer  = theTriggerAnalyzers.begin(); 
	thisAnalyzer != theTriggerAnalyzers.end(); ++thisAnalyzer )
    {
      (*thisAnalyzer)->analyze(iEvent);
    } 
  theTimeAnalyzer    ->analyze(iEvent);
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
  theTimeAnalyzer    ->begin();
  theOverlapAnalyzer ->begin();
}



void 
MuonTriggerRateTimeAnalyzer::endJob() {
  using namespace edm;
  vector<HLTMuonGenericRate *>::iterator thisAnalyzer;
  for ( thisAnalyzer  = theTriggerAnalyzers.begin(); 
        thisAnalyzer != theTriggerAnalyzers.end(); 
	++thisAnalyzer )
    {
      (*thisAnalyzer)->finish();
    }
  theTimeAnalyzer    ->finish();
  theOverlapAnalyzer ->finish();
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonTriggerRateTimeAnalyzer);
