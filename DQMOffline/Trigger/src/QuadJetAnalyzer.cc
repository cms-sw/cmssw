// -*- C++ -*-
//
// Package:     QuadJetAnalyzer
// Class:       QuadJetAnalyzer
// 
/**\class MuonTriggerRateTimeAnalyzer MuonTriggerRateTimeAnalyzer.cc HLTriggerOffline/Muon/src/MuonTriggerRateTimeAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Muriel Vander Donckt
//         Created:  Tue Jul 24 12:17:12 CEST 2007
// $Id: QuadJetAnalyzer.cc,v 1.1 2009/10/09 12:49:43 slaunwhj Exp $
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

//#include "DQMOffline/Trigger/interface/HLTMuonMatchAndPlot.h"
//#include "DQMOffline/Trigger/interface/HLTMuonOverlap.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
//#include "PhysicsTools/Utilities/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

#include "TFile.h"
#include "TDirectory.h"





class QuadJetAnalyzer : public edm::EDAnalyzer {

public:
  explicit QuadJetAnalyzer(const edm::ParameterSet&);
  ~QuadJetAnalyzer();

private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  int theNumberOfTriggers;

  bool atLeastOneValidTrigger;

  DQMStore* dbe_;
  MonitorElement * testHist;
};

using namespace std;
using namespace edm;
//using reco::Muon;


QuadJetAnalyzer::QuadJetAnalyzer(const ParameterSet& pset)
{

  LogTrace ("HLTMuonVal") << "\n\n Inside MuonTriggerRate Constructor\n\n";
  
  vector<string> triggerNames = pset.getParameter< vector<string> >
                                ("TriggerNames");
  string theHltProcessName = pset.getParameter<string>("HltProcessName");

  //vector<edm::ParameterSet> customCollection = pset.getParameter<vector<edm::ParameterSet> > ("customCollection");
  //vector<edm::ParameterSet>::iterator iPSet;


    
  dbe_ = 0;
  if ( pset.getUntrackedParameter<bool>("DQMStore", false) ) {
    dbe_ = Service<DQMStore>().operator->();
    dbe_->setVerbose(3);    
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


}


QuadJetAnalyzer::~QuadJetAnalyzer()
{

  LogTrace ("HLTMuonVal")
    << "Inside QuadJetAnalyzer destructor" << endl;

}


//
// member functions
//

void
QuadJetAnalyzer::analyze(const Event& iEvent, const EventSetup& iSetup)
{
  LogTrace ("HLTMuonVal")
    << "Inside QuadJetAnalyzer analyze" << endl;

  bool isFired = false;

  edm::Handle<edm::TriggerResults> HLTR;
  iEvent.getByLabel(edm::InputTag("TriggerResults::HLT"), HLTR); 
  if(!HLTR.isValid()) {

    iEvent.getByLabel(edm::InputTag("TriggerResults::FU"), HLTR); 

    if(!HLTR.isValid()) {
      edm::LogInfo("FourVectorHLTOffline") << "TriggerResults not found, "
      "skipping event"; 
      return;
   }
  }

  HLTConfigProvider hltConfig;
  hltConfig.init("HLT");

  LogTrace ("HLTMuonVal")
    << " hltConfig.size() " << hltConfig.size() << std::endl;
  unsigned int triggerIndex( hltConfig.triggerIndex("HLT_QuadJet30") );
  
  // triggerIndex must be less than the size of HLTR or you get a CMSException: _M_range_check
  if (triggerIndex < HLTR->size()) isFired = HLTR->accept(triggerIndex);
   
  //std::cout << "*******FIRED**********" << isFired << std::endl;


  edm::Handle<trigger::TriggerEvent> triggerEvent_;
  iEvent.getByLabel("hltTriggerSummaryAOD", triggerEvent_); 

  if(!triggerEvent_.isValid()) {

    edm::LogInfo("FourVectorHLTOffline") << "TriggerSummaryAOD not found, "
      "skipping event"; 
    return;

  }

  testHist->Fill(1);


  //// Temporary debug
  /*
  for (int i = 0; i < triggerEvent_->sizeFilters(); ++i){
    std::cout << "filter Tag [" << i << "] = " << triggerEvent_->filterTag(i) << std::endl;
  }
  const size_t nF(triggerEvent_->sizeFilters());
  std::cout << "Number of TriggerFilters: " << nF << std::endl;
  std::cout << "The Filters: #, tag, #ids/#keys, the id/key pairs" << std::endl;
  for (size_t iF=0; iF!=nF; ++iF) {
    const trigger::Vids& VIDS(triggerEvent_->filterIds(iF));
    const trigger::Keys& KEYS(triggerEvent_->filterKeys(iF));
    const size_t nI(VIDS.size());
    const size_t nK(KEYS.size());
    std::cout << iF << " " << triggerEvent_->filterTag(iF).encode()
	      << " " << nI << "/" << nK
	      << " the pairs: ";
    const size_t n(std::max(nI,nK));
    for (size_t i=0; i!=n; ++i) {
      std::cout << " " << VIDS[i] << "/" << KEYS[i];
    }
    std::cout << std::endl;
  }
  */
  //// Temporary debug


}



void 
QuadJetAnalyzer::beginJob()
{

  LogTrace ("HLTMuonVal")
    << "Inside QuadJetAnalyzer beginJob" << endl;

  testHist =dbe_->book1D("testHist","testHist",5,-0.5,4.5);

}



void 
QuadJetAnalyzer::endJob() {

  LogTrace ("HLTMuonVal")
    << "Inside QuadJetAnalyzer endJob()" << endl;
  

}

//define this as a plug-in
DEFINE_FWK_MODULE(QuadJetAnalyzer);
