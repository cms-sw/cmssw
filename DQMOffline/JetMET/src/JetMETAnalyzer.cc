/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/12/08 12:06:28 $
 *  $Revision: 1.18 $
 *  \author F. Chlebana - Fermilab
 *          K. Hatakeyama - Rockefeller University
 */

#include "DQMOffline/JetMET/src/JetMETAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
using namespace std;
using namespace edm;

#define DEBUG 0

// ***********************************************************
JetMETAnalyzer::JetMETAnalyzer(const edm::ParameterSet& pSet) {

  parameters = pSet;
  
  // Calo Jet Collection Label
  theSCJetCollectionLabel       = parameters.getParameter<edm::InputTag>("SCJetsCollectionLabel");
  theICJetCollectionLabel       = parameters.getParameter<edm::InputTag>("ICJetsCollectionLabel");
  theJPTJetCollectionLabel      = parameters.getParameter<edm::InputTag>("JPTJetsCollectionLabel");

  thePFJetCollectionLabel       = parameters.getParameter<edm::InputTag>("PFJetsCollectionLabel");

  theCaloMETCollectionLabel     = parameters.getParameter<edm::InputTag>("CaloMETCollectionLabel");
  theCaloMETNoHFCollectionLabel = parameters.getParameter<edm::InputTag>("CaloMETNoHFCollectionLabel");

  theTriggerResultsLabel        = parameters.getParameter<edm::InputTag>("TriggerResultsLabel");
  
  theJetAnalyzerFlag            = parameters.getUntrackedParameter<bool>("DoJetAnalysis",    true);
  theJPTJetAnalyzerFlag         = parameters.getUntrackedParameter<bool>("DoJPTJetAnalysis", true);
  thePFJetAnalyzerFlag          = parameters.getUntrackedParameter<bool>("DoPFJetAnalysis",  true);
  theCaloMETAnalyzerFlag        = parameters.getUntrackedParameter<bool>("DoCaloMETAnalysis",true);

  // --- do the analysis on the Jets
  if(theJetAnalyzerFlag) {
    theSCJetAnalyzer  = new JetAnalyzer(parameters.getParameter<ParameterSet>("jetAnalysis"));
    theSCJetAnalyzer->setSource("SISConeJets");
    theICJetAnalyzer  = new JetAnalyzer(parameters.getParameter<ParameterSet>("jetAnalysis"));
    theICJetAnalyzer->setSource("IterativeConeJets");
  }

  // --- do the analysis on JPT Jets
  if(theJPTJetAnalyzerFlag) {
    theJPTJetAnalyzer  = new JetAnalyzer(parameters.getParameter<ParameterSet>("JPTJetAnalysis"));
    theJPTJetAnalyzer->setSource("JPTJets");
  }

  // --- do the analysis on the PFJets
  if(thePFJetAnalyzerFlag)
    thePFJetAnalyzer = new PFJetAnalyzer(parameters.getParameter<ParameterSet>("pfJetAnalysis"));

  // --- do the analysis on the MET
  if(theCaloMETAnalyzerFlag)
    theCaloMETAnalyzer = new CaloMETAnalyzer(parameters.getParameter<ParameterSet>("caloMETAnalysis"));

  LoJetTrigger = parameters.getParameter<std::string>("JetLo");
  HiJetTrigger = parameters.getParameter<std::string>("JetHi");

}

// ***********************************************************
JetMETAnalyzer::~JetMETAnalyzer() {   
  if(theJetAnalyzerFlag) {
    delete theSCJetAnalyzer;
    delete theICJetAnalyzer;
  }
  if(theJPTJetAnalyzerFlag)  delete theJPTJetAnalyzer;
  if(thePFJetAnalyzerFlag)   delete thePFJetAnalyzer;
  if(theCaloMETAnalyzerFlag) delete theCaloMETAnalyzer;
}

// ***********************************************************
void JetMETAnalyzer::beginJob(edm::EventSetup const& iSetup) {

  metname = "JetMETAnalyzer";

  LogTrace(metname)<<"[JetMETAnalyzer] Parameters initialization";
  dbe = edm::Service<DQMStore>().operator->();

  if(theJetAnalyzerFlag) { 
    theSCJetAnalyzer->beginJob(iSetup, dbe);
    theICJetAnalyzer->beginJob(iSetup, dbe);
  }
  if(theJPTJetAnalyzerFlag)  theJPTJetAnalyzer->beginJob(iSetup, dbe);
  if(thePFJetAnalyzerFlag)   thePFJetAnalyzer->beginJob(iSetup, dbe);
  if(theCaloMETAnalyzerFlag) theCaloMETAnalyzer->beginJob(iSetup, dbe);

}


// ***********************************************************
void JetMETAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  LogTrace(metname)<<"[JetMETAnalyzer] Analysis of event # ";

  // **** Get the TriggerResults container
  edm::Handle<TriggerResults> triggerResults;
  iEvent.getByLabel(theTriggerResultsLabel, triggerResults);

  if (DEBUG)  std::cout << "trigger label " << theTriggerResultsLabel << std::endl;

  Int_t JetLoPass = 0;
  Int_t JetHiPass = 0;

  if (triggerResults.isValid()) {

    if (DEBUG) std::cout << "trigger valid " << std::endl;
    edm::TriggerNames triggerNames;    // TriggerNames class
    triggerNames.init(*triggerResults);
    unsigned int n = triggerResults->size();
    for (unsigned int i=0; i!=n; i++) {

      if ( triggerNames.triggerName(i) == LoJetTrigger ) {
	JetLoPass =  triggerResults->accept(i);
	if (DEBUG) std::cout << "Found  HLT_Jet30" << std::endl;
      }
      if ( triggerNames.triggerName(i) == HiJetTrigger ) {
	JetHiPass =  triggerResults->accept(i);
      }
    }

  } else {

    //
    //edm::Handle<TriggerResults> *tr = new edm::Handle<TriggerResults>;
    //triggerResults = (*tr);
    triggerResults = edm::Handle<TriggerResults>(); 

    if (DEBUG) std::cout << "trigger not valid " << std::endl;
    edm::LogInfo("JetMETAnalyzer") << "TriggerResults::HLT not found, "
      "automatically select events";

  }

  if (DEBUG) {
    std::cout << ">>> Trigger  Lo = " <<  JetLoPass
	      <<             " Hi = " <<  JetHiPass
	      << std::endl;
  }


  // **** Get the Calo Jet container
  edm::Handle<reco::CaloJetCollection> caloJets;

  // **** Get the SISCone Jet container
  iEvent.getByLabel(theSCJetCollectionLabel, caloJets);

  if(caloJets.isValid()){
    theSCJetAnalyzer->setJetHiPass(JetHiPass);
    theSCJetAnalyzer->setJetLoPass(JetLoPass);
    for (reco::CaloJetCollection::const_iterator cal = caloJets->begin(); cal!=caloJets->end(); ++cal){
      if(theJetAnalyzerFlag){
	LogTrace(metname)<<"[JetMETAnalyzer] Call to the SC Jet analyzer";
	if (cal == caloJets->begin()) {	  
	  theSCJetAnalyzer->setNJets(caloJets->size());
	  theSCJetAnalyzer->setLeadJetFlag(1);
	} else {
	  theSCJetAnalyzer->setLeadJetFlag(0);
	}
	theSCJetAnalyzer->analyze(iEvent, iSetup, *cal);
      }
    }
  }

  // **** Get the Iterative Cone Jet container
  iEvent.getByLabel(theICJetCollectionLabel, caloJets);

  if(caloJets.isValid()){
    theICJetAnalyzer->setJetHiPass(JetHiPass);
    theICJetAnalyzer->setJetLoPass(JetLoPass);
    for (reco::CaloJetCollection::const_iterator cal = caloJets->begin(); cal!=caloJets->end(); ++cal){
      if(theJetAnalyzerFlag){
	LogTrace(metname)<<"[JetMETAnalyzer] Call to the IC Jet analyzer";
	if (cal == caloJets->begin()) {
	  theICJetAnalyzer->setNJets(caloJets->size());
	  theICJetAnalyzer->setLeadJetFlag(1);
	} else {
	  theICJetAnalyzer->setLeadJetFlag(0);
	}
	//	theICJetAnalyzer->analyze(iEvent, iSetup, *triggerResults, *cal);	
	theICJetAnalyzer->analyze(iEvent, iSetup, *cal);	
      }
    }
  }

  // **** Get the JPT Jet container
  iEvent.getByLabel(theJPTJetCollectionLabel, caloJets);
  //jpt
  if(caloJets.isValid()){
    theJPTJetAnalyzer->setJetHiPass(JetHiPass);
    theJPTJetAnalyzer->setJetLoPass(JetLoPass);
    for (reco::CaloJetCollection::const_iterator cal = caloJets->begin(); cal!=caloJets->end(); ++cal){
      if(theJetAnalyzerFlag){
	LogTrace(metname)<<"[JetMETAnalyzer] Call to the JPT Jet analyzer";
	if (cal == caloJets->begin()) {	  
	  theJPTJetAnalyzer->setNJets(caloJets->size());
	  theJPTJetAnalyzer->setLeadJetFlag(1);
	} else {
	  theJPTJetAnalyzer->setLeadJetFlag(0);
	}
	theJPTJetAnalyzer->analyze(iEvent, iSetup, *cal);
      }
    }
  }

  // **** Get the PFlow Jet container
  edm::Handle<reco::PFJetCollection> pfJets;
  iEvent.getByLabel(thePFJetCollectionLabel, pfJets);

  if(pfJets.isValid()){
//     for (reco::PFJetCollection::const_iterator cal = pfJets->begin(); cal!=pfJets->end(); ++cal){
//       if(thePFJetAnalyzerFlag){
// 	LogTrace(metname)<<"[JetMETAnalyzer] Call to the PFJet analyzer";
// 	thePFJetAnalyzer->analyze(iEvent, iSetup, *cal);
//       }
//     }
    thePFJetAnalyzer->setJetHiPass(JetHiPass);
    thePFJetAnalyzer->setJetLoPass(JetLoPass);
    for (reco::PFJetCollection::const_iterator cal = pfJets->begin(); cal!=pfJets->end(); ++cal){
      if(thePFJetAnalyzerFlag){
	if (cal == pfJets->begin()) {	  
	  thePFJetAnalyzer->setLeadJetFlag(1);
	} else {
	  thePFJetAnalyzer->setLeadJetFlag(0);
	}
	LogTrace(metname)<<"[JetMETAnalyzer] Call to the PFJet analyzer";
	thePFJetAnalyzer->analyze(iEvent, iSetup, *cal);
      }
    }
  } else {
    if (DEBUG) LogTrace(metname)<<"[JetMETAnalyzer] pfjets NOT VALID!!";
  }

  // **** Get the MET container  
  edm::Handle<reco::CaloMETCollection> calometcoll;
  iEvent.getByLabel(theCaloMETCollectionLabel, calometcoll);
  edm::Handle<reco::CaloMETCollection> calometNoHFcoll;
  iEvent.getByLabel(theCaloMETNoHFCollectionLabel, calometNoHFcoll);

  if(calometcoll.isValid() && calometNoHFcoll.isValid()){
    const CaloMETCollection *calometcol = calometcoll.product();
    const CaloMET *calomet;
    calomet = &(calometcol->front());
    const CaloMETCollection *calometNoHFcol = calometNoHFcoll.product();
    const CaloMET *calometNoHF;
    calometNoHF = &(calometNoHFcol->front());

    if(theCaloMETAnalyzerFlag){
      LogTrace(metname)<<"[JetMETAnalyzer] Call to the CaloMET analyzer";
      theCaloMETAnalyzer->analyze(iEvent, iSetup,
				  *triggerResults,
				  *calomet, 
				  *calometNoHF);
    }
  }

}

// ***********************************************************
void JetMETAnalyzer::endJob(void) {
  LogTrace(metname)<<"[JetMETAnalyzer] Saving the histos";
  bool outputMEsInRootFile   = parameters.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName = parameters.getParameter<std::string>("OutputFileName");

  if(outputMEsInRootFile){
    dbe->showDirStructure();
    dbe->save(outputFileName);
  }
}

