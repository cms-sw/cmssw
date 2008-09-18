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
// $Id: MuonTriggerRateTimeAnalyzer.cc,v 1.7 2008/09/18 18:52:08 klukas Exp $
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
      // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//
typedef std::vector< edm::ParameterSet > Parameters;

//
// constructors and destructor
//

MuonTriggerRateTimeAnalyzer::MuonTriggerRateTimeAnalyzer(const edm::ParameterSet& pset)
{
    Parameters TriggerLists = pset.getParameter<Parameters>
                              ("TriggerCollection");
    theNumberOfTriggers     = TriggerLists.size();
    for( int i = 0; i < theNumberOfTriggers; i++) {
      HLTMuonGenericRate *hmg = new HLTMuonGenericRate( pset, i );
      theTriggerAnalyzers.push_back( hmg );
    }
    theOverlapAnalyzer = new HLTMuonOverlap(pset);    
    theTimeAnalyzer    = new HLTMuonTime(pset);    
}


MuonTriggerRateTimeAnalyzer::~MuonTriggerRateTimeAnalyzer()
{
  using namespace edm;
  std::vector<HLTMuonGenericRate *>::iterator iTrig;
  for ( iTrig  = theTriggerAnalyzers.begin(); 
        iTrig != theTriggerAnalyzers.end(); ++iTrig )
  {
    delete *iTrig;
  } 
  theTriggerAnalyzers.clear();
  delete theOverlapAnalyzer;
  delete theTimeAnalyzer;

}


//
// member functions
//

void
MuonTriggerRateTimeAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
  std::vector<HLTMuonGenericRate *>::iterator iTrig;
   for ( iTrig  = theTriggerAnalyzers.begin(); 
	 iTrig != theTriggerAnalyzers.end(); ++iTrig )
   {
     (*iTrig)->analyze(iEvent);
   } 
   theTimeAnalyzer   ->analyze(iEvent);
   theOverlapAnalyzer->analyze(iEvent);
}



void 
MuonTriggerRateTimeAnalyzer::beginJob(const edm::EventSetup&)
{
  std::vector<HLTMuonGenericRate *>::iterator iTrig;
  for ( iTrig  = theTriggerAnalyzers.begin(); 
        iTrig != theTriggerAnalyzers.end(); ++iTrig )
  {
    (*iTrig)->begin();
  } 
  theTimeAnalyzer   ->begin();
  theOverlapAnalyzer->begin();
}



void 
MuonTriggerRateTimeAnalyzer::endJob() {
  using namespace edm;
  std::vector<HLTMuonGenericRate *>::iterator iTrig;
  for ( iTrig  = theTriggerAnalyzers.begin(); 
        iTrig != theTriggerAnalyzers.end(); ++iTrig )
  {
    (*iTrig)->finish();
  }
  theTimeAnalyzer   ->finish();
  theOverlapAnalyzer->finish();
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonTriggerRateTimeAnalyzer);
