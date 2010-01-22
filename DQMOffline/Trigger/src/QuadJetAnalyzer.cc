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
// $Id: QuadJetAnalyzer.cc,v 1.6 2010/01/12 14:11:20 dellaric Exp $
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
#include "DataFormats/Common/interface/Handle.h"

//#include "DQMOffline/Trigger/interface/HLTMuonMatchAndPlot.h"
//#include "DQMOffline/Trigger/interface/HLTMuonOverlap.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
//#include "CommonTools/Utilities/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/Common/interface/TriggerResults.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"

#include "TFile.h"
#include "TDirectory.h"


using namespace std;
using namespace edm;
using namespace reco;
using namespace trigger;
using namespace l1extra;



class QuadJetAnalyzer : public edm::EDAnalyzer {

public:
  explicit QuadJetAnalyzer(const edm::ParameterSet&);
  ~QuadJetAnalyzer();

private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  void sortJets (std::vector<const trigger::TriggerObject*> theJets);

  int theNumberOfTriggers;

  string theHltProcessName;
  
  bool atLeastOneValidTrigger;

  DQMStore* dbe_;
  MonitorElement * trigjets_n;
  MonitorElement * trigjet_et_hi;
  MonitorElement * trigjet_et_lo;
  MonitorElement * trigjet_etaphi_lo;
  MonitorElement * trigjet_etaphi_hi;
};



QuadJetAnalyzer::QuadJetAnalyzer(const ParameterSet& pset)
{

  LogTrace ("HLTMuonVal") << "\n\n Inside MuonTriggerRate Constructor\n\n";
  
  vector<string> triggerNames = pset.getParameter< vector<string> >
                                ("TriggerNames");
  theHltProcessName = pset.getParameter<string>("HltProcessName");

  //vector<edm::ParameterSet> customCollection = pset.getParameter<vector<edm::ParameterSet> > ("customCollection");
  //vector<edm::ParameterSet>::iterator iPSet;


    
  dbe_ = 0;
  if ( pset.getUntrackedParameter<bool>("DQMStore", false) ) {
    dbe_ = Service<DQMStore>().operator->();
    dbe_->setVerbose(3);    
  }
  
  
  LogTrace ("HLTMuonVal") << "Initializing HLTConfigProvider with HLT process name: " << theHltProcessName << endl;
  HLTConfigProvider hltConfig;
  bool hltConfigInitSuccess = hltConfig.init(theHltProcessName);
  vector<string> validTriggerNames;

  if (hltConfigInitSuccess)
    validTriggerNames = hltConfig.triggerNames();

  if (!hltConfigInitSuccess ) {
    LogInfo ("HLTMuonVal") << endl << endl << endl
                           << "---> WARNING: The HLT Config Provider gave you an empty list of valid trigger names" << endl
                           << "Could be a problem with the HLT Process Name (you provided  " << theHltProcessName <<")" << endl
                           << "W/o valid triggers we can't produce plots, exiting..."
                           << endl << endl << endl;
    
  } else {

    // you successfully initialized hlt config provider
    // do hlt config dependent processing

    
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


  Handle<CaloJetCollection> jetsHandle;
  iEvent.getByLabel("iterativeCone5CaloJets", jetsHandle);

  edm::Handle<edm::TriggerResults> HLTR;
  iEvent.getByLabel(edm::InputTag("TriggerResults::HLT"), HLTR); 

  HLTConfigProvider hltConfig;
  bool hltConfigInitSuccess = hltConfig.init(theHltProcessName);

  if (!hltConfigInitSuccess) {
    LogTrace("HLTMuonVal") << "no valid hltConfigProvider, stop analyzing thisevent" << endl;

    return;
  }
  
  LogTrace ("HLTMuonVal")
    << " hltConfig.size() " << hltConfig.size() << std::endl;
  unsigned int triggerIndex( hltConfig.triggerIndex("HLT_QuadJet30") );

  bool isFired = false;  
  // triggerIndex must be less than the size of HLTR or you get a CMSException: _M_range_check
  if (triggerIndex < HLTR->size()) isFired = HLTR->accept(triggerIndex);
   
  if(isFired==false)return;

  int nqj30jets = 0;

  std::vector<trigger::TriggerObject> corrTriggerJets;

  edm::Handle<trigger::TriggerEvent> trgEvent;
  iEvent.getByLabel(InputTag("hltTriggerSummaryAOD","","HLT"), trgEvent);
  if(!trgEvent.isValid()) {

    LogInfo ("HLTMuonVal") << "TriggerSummaryAOD not found, skipping event" << endl;
    return;
  }

  trigger::size_type triggerKey = trgEvent->filterIndex(edm::InputTag("hlt4jet30", "", "HLT"));
  
  const TriggerObjectCollection& TOC(trgEvent->getObjects());
  // filterIndex must be less than the size of trgEvent or you get a CMSException: _M_range_check
  if ( triggerKey < trgEvent->sizeFilters() ) {
    const Keys& keys( trgEvent->filterKeys(triggerKey) );
    
    for (size_t hlto = 0; hlto < keys.size(); hlto++ ) {
      size_type hltf = keys[hlto];
      TriggerObject L3obj(TOC[hltf]);
      corrTriggerJets.push_back(L3obj);
      nqj30jets++;
    }
  }
  

  ////SORT THE JETS BY ET

  for ( size_t iJet = 0; iJet < corrTriggerJets.size(); iJet ++) {
    for ( size_t jJet = iJet; jJet < corrTriggerJets.size(); jJet ++) {
      
      if ( corrTriggerJets[jJet].et() > corrTriggerJets[iJet].et() ) {
        const trigger::TriggerObject tmpJet =  corrTriggerJets[iJet];
        corrTriggerJets[iJet] = corrTriggerJets[jJet];
        corrTriggerJets[jJet] = tmpJet;        
      }
    }// end for each jJet
  }// end for each iJet
  

  if(corrTriggerJets.size()> 0){
    trigjet_etaphi_hi->Fill(corrTriggerJets[0].eta(),corrTriggerJets[0].phi());
    trigjet_etaphi_lo->Fill(corrTriggerJets[nqj30jets-1].eta(),corrTriggerJets[nqj30jets-1].phi());
    trigjet_et_hi->Fill(corrTriggerJets[0].et());
    trigjet_et_lo->Fill(corrTriggerJets[nqj30jets-1].et());
    trigjets_n->Fill(nqj30jets);
  }
}



void 
QuadJetAnalyzer::beginJob()
{

  LogTrace ("HLTMuonVal")
    << "Inside QuadJetAnalyzer beginJob" << endl;

  dbe_->setCurrentFolder("HLT/JetMET/QuadJet");
  
  double PI = 3.14159;
  trigjets_n =dbe_->book1D("trigjets_n","trigjets_n",21,-0.5,20.5);
  trigjet_et_lo =dbe_->book1D("trigjet_et_lo","trigjet_et_lo",50,0,500);
  trigjet_et_hi =dbe_->book1D("trigjet_et_hi","trigjet_et_hi",50,0,500);
  trigjet_etaphi_lo =dbe_->book2D("trigjet_etaphi_lo","trigjet_etaphi_lo",50,-5.,5.,24,-PI,PI);
  trigjet_etaphi_hi =dbe_->book2D("trigjet_etaphi_hi","trigjet_etaphi_hi",50,-5.,5.,24,-PI,PI);

}



void 
QuadJetAnalyzer::endJob() {

  LogTrace ("HLTMuonVal")
    << "Inside QuadJetAnalyzer endJob()" << endl;
  

}

//define this as a plug-in
DEFINE_FWK_MODULE(QuadJetAnalyzer);
