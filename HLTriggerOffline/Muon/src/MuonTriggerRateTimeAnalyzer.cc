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
// $Id: MuonTriggerRateTimeAnalyzer.cc,v 1.5 2008/07/22 09:36:28 klukas Exp $
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
//
// class declaration
//

class MuonTriggerRateTimeAnalyzer : public edm::EDAnalyzer {
   public:
      explicit MuonTriggerRateTimeAnalyzer(const edm::ParameterSet&);
      ~MuonTriggerRateTimeAnalyzer();
   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
  int NumberOfTriggers;
  std::vector<HLTMuonGenericRate *> muTriggerAnalyzer;
  HLTMuonOverlap *OverlapAnalyzer;
  HLTMuonTime *TimeAnalyzer;
      // ----------member data ---------------------------
};

//
// constants, enums and typedefs
//
typedef std::vector< edm::ParameterSet > Parameters;

//
// static data member definitions
//

//
// constructors and destructor
//
MuonTriggerRateTimeAnalyzer::MuonTriggerRateTimeAnalyzer(const edm::ParameterSet& pset)
{
   //now do what ever initialization is needed
  // edm::ParameterSet* SingleMuSet=new edm::ParameterSet(pset);
    Parameters TriggerLists=pset.getParameter<Parameters>("TriggerCollection");
    NumberOfTriggers=TriggerLists.size();
    for( int index=0; index < NumberOfTriggers; index++) {
      HLTMuonGenericRate *hmg=new HLTMuonGenericRate(pset,index);
      muTriggerAnalyzer.push_back(hmg);
    }
    OverlapAnalyzer=new HLTMuonOverlap(pset);    
    TimeAnalyzer=new HLTMuonTime(pset);    
}


MuonTriggerRateTimeAnalyzer::~MuonTriggerRateTimeAnalyzer()
{
  using namespace edm;
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
  for (  std::vector<HLTMuonGenericRate *>::iterator iTrig = muTriggerAnalyzer.begin(); iTrig != muTriggerAnalyzer.end(); ++iTrig){
    delete *iTrig;
  } 
  muTriggerAnalyzer.clear();
  delete OverlapAnalyzer;
  delete TimeAnalyzer;

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
MuonTriggerRateTimeAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   for (  std::vector<HLTMuonGenericRate *>::iterator iTrig = muTriggerAnalyzer.begin(); iTrig != muTriggerAnalyzer.end(); ++iTrig){
     (*iTrig)->analyze(iEvent);
   } 
   OverlapAnalyzer->analyze(iEvent);
   TimeAnalyzer->analyze(iEvent);
}


// ------------ method called once each job just before starting event loop  ------------
void 
MuonTriggerRateTimeAnalyzer::beginJob(const edm::EventSetup&)
{

  for (  std::vector<HLTMuonGenericRate *>::iterator iTrig = muTriggerAnalyzer.begin(); iTrig != muTriggerAnalyzer.end(); ++iTrig){
    (*iTrig)->BookHistograms();
  } 
  TimeAnalyzer->BookHistograms();
}

// ------------ method called once each job just after ending the event loop  ------------
void 
MuonTriggerRateTimeAnalyzer::endJob() {
  using namespace edm;
  TimeAnalyzer->WriteHistograms();
  OverlapAnalyzer->getResults();
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonTriggerRateTimeAnalyzer);
