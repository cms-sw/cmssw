/*
 *  See header file for a description of this class.
 *
 *  $Date: 2009/12/06 10:10:27 $
 *  $Revision: 1.37 $
 *  \author F. Chlebana - Fermilab
 *          K. Hatakeyama - Rockefeller University
 */

#include "DQMOffline/JetMET/interface/JetMETAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

//#include "DataFormats/METReco/interface/CaloMETCollection.h"
//#include "DataFormats/METReco/interface/CaloMET.h"
//#include "DataFormats/METReco/interface/METCollection.h"
//#include "DataFormats/METReco/interface/MET.h"
//#include "DataFormats/METReco/interface/PFMETCollection.h"
//#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/CaloMETFwd.h"
#include "DataFormats/METReco/interface/METFwd.h"
#include "DataFormats/METReco/interface/PFMETFwd.h"
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
  
  // Jet Collection Label 
  theAKJetCollectionLabel       = parameters.getParameter<edm::InputTag>("AKJetsCollectionLabel");
  theSCJetCollectionLabel       = parameters.getParameter<edm::InputTag>("SCJetsCollectionLabel");
  theICJetCollectionLabel       = parameters.getParameter<edm::InputTag>("ICJetsCollectionLabel");
  theJPTJetCollectionLabel      = parameters.getParameter<edm::InputTag>("JPTJetsCollectionLabel");
  thePFJetCollectionLabel       = parameters.getParameter<edm::InputTag>("PFJetsCollectionLabel");

  theTriggerResultsLabel        = parameters.getParameter<edm::InputTag>("TriggerResultsLabel");
  
  theJetAnalyzerFlag            = parameters.getUntrackedParameter<bool>("DoJetAnalysis",    true);
  theJetCleaningFlag            = parameters.getUntrackedParameter<bool>("DoJetCleaning",    true);
  theIConeJetAnalyzerFlag       = parameters.getUntrackedParameter<bool>("DoIterativeCone",  false);
  theJetPtAnalyzerFlag          = parameters.getUntrackedParameter<bool>("DoJetPtAnalysis",  false);
  theJPTJetAnalyzerFlag         = parameters.getUntrackedParameter<bool>("DoJPTJetAnalysis", true);
  thePFJetAnalyzerFlag          = parameters.getUntrackedParameter<bool>("DoPFJetAnalysis",  true);
  theCaloMETAnalyzerFlag        = parameters.getUntrackedParameter<bool>("DoCaloMETAnalysis",true);
  theTcMETAnalyzerFlag          = parameters.getUntrackedParameter<bool>("DoTcMETAnalysis",  true);
  theMuCorrMETAnalyzerFlag      = parameters.getUntrackedParameter<bool>("DoMuCorrMETAnalysis",  true);
  thePfMETAnalyzerFlag          = parameters.getUntrackedParameter<bool>("DoPfMETAnalysis",  true);
  theHTMHTAnalyzerFlag          = parameters.getUntrackedParameter<bool>("DoHTMHTAnalysis",  true);

  // --- do the analysis on the Jets
  if(theJetAnalyzerFlag) {
    theSCJetAnalyzer  = new JetAnalyzer(parameters.getParameter<ParameterSet>("jetAnalysis"));
    theSCJetAnalyzer->setSource("SISConeJets");
    theAKJetAnalyzer  = new JetAnalyzer(parameters.getParameter<ParameterSet>("jetAnalysis"));
    theAKJetAnalyzer->setSource("AntiKtJets");    
    if(theIConeJetAnalyzerFlag){
    theICJetAnalyzer  = new JetAnalyzer(parameters.getParameter<ParameterSet>("jetAnalysis"));
    theICJetAnalyzer->setSource("IterativeConeJets");  
    }
  }

  if(theJetCleaningFlag) {
    theCleanedSCJetAnalyzer  = new JetAnalyzer(parameters.getParameter<ParameterSet>("CleanedjetAnalysis"));
    theCleanedSCJetAnalyzer->setSource("CleanedSISConeJets");
    theCleanedAKJetAnalyzer  = new JetAnalyzer(parameters.getParameter<ParameterSet>("CleanedjetAnalysis"));
    theCleanedAKJetAnalyzer->setSource("CleanedAntiKtJets");
    if(theIConeJetAnalyzerFlag){
    theCleanedICJetAnalyzer  = new JetAnalyzer(parameters.getParameter<ParameterSet>("CleanedjetAnalysis"));
    theCleanedICJetAnalyzer->setSource("CleanedIterativeConeJets");
    }
  }
  // Do Pt analysis
  if(theJetPtAnalyzerFlag ) {
    thePtAKJetAnalyzer  = new JetPtAnalyzer(parameters.getParameter<ParameterSet>("PtAnalysis"));
    thePtAKJetAnalyzer->setSource("PtAnalysisAntiKtJets");
    thePtSCJetAnalyzer  = new JetPtAnalyzer(parameters.getParameter<ParameterSet>("PtAnalysis"));
    thePtSCJetAnalyzer->setSource("PtAnalysisSISConeJets");
    thePtICJetAnalyzer  = new JetPtAnalyzer(parameters.getParameter<ParameterSet>("PtAnalysis"));
    thePtICJetAnalyzer->setSource("PtAnalysisIterativeConeJets");
  }

  // --- do the analysis on JPT Jets
  if(theJPTJetAnalyzerFlag) {
    //theJPTJetAnalyzer  = new JetAnalyzer(parameters.getParameter<ParameterSet>("JPTJetAnalysis"));
    //theJPTJetAnalyzer->setSource("JPTJets");
    theJPTJetAnalyzer  = new JPTJetAnalyzer(parameters.getParameter<ParameterSet>("JPTJetAnalysis"));
  }

  // --- do the analysis on the PFJets
  if(thePFJetAnalyzerFlag)
    thePFJetAnalyzer = new PFJetAnalyzer(parameters.getParameter<ParameterSet>("pfJetAnalysis"));

  LoJetTrigger = parameters.getParameter<std::string>("JetLo");
  HiJetTrigger = parameters.getParameter<std::string>("JetHi");

  // --- do the analysis on the MET
  if(theCaloMETAnalyzerFlag){
     theCaloMETAnalyzer       = new CaloMETAnalyzer(parameters.getParameter<ParameterSet>("caloMETAnalysis"));
     theCaloMETNoHFAnalyzer   = new CaloMETAnalyzer(parameters.getParameter<ParameterSet>("caloMETNoHFAnalysis"));
     theCaloMETHOAnalyzer     = new CaloMETAnalyzer(parameters.getParameter<ParameterSet>("caloMETHOAnalysis"));
     theCaloMETNoHFHOAnalyzer = new CaloMETAnalyzer(parameters.getParameter<ParameterSet>("caloMETNoHFHOAnalysis"));
  }
  if(theTcMETAnalyzerFlag){
     theTcMETAnalyzer = new METAnalyzer(parameters.getParameter<ParameterSet>("tcMETAnalysis"));
  }
  if(theMuCorrMETAnalyzerFlag){
     theMuCorrMETAnalyzer = new METAnalyzer(parameters.getParameter<ParameterSet>("mucorrMETAnalysis"));
  }
  if(thePfMETAnalyzerFlag){
     thePfMETAnalyzer = new PFMETAnalyzer(parameters.getParameter<ParameterSet>("pfMETAnalysis"));
  }
  if(theHTMHTAnalyzerFlag){
     theHTMHTAnalyzer         = new HTMHTAnalyzer(parameters.getParameter<ParameterSet>("HTMHTAnalysis"));
  }

  _LSBegin     = parameters.getParameter<int>("LSBegin");
  _LSEnd       = parameters.getParameter<int>("LSEnd");

  processname_ = parameters.getParameter<std::string>("processname");

}

// ***********************************************************
JetMETAnalyzer::~JetMETAnalyzer() {   

  if(theJetAnalyzerFlag) {
    delete theSCJetAnalyzer;    
    delete theAKJetAnalyzer;
    if(theIConeJetAnalyzerFlag) delete theICJetAnalyzer;
  }
  if(theJetCleaningFlag) {
    delete theCleanedSCJetAnalyzer;
    delete theCleanedAKJetAnalyzer;
    if(theIConeJetAnalyzerFlag) delete theCleanedICJetAnalyzer;
  }
  if(theJetPtAnalyzerFlag) {
    delete thePtSCJetAnalyzer;
    delete thePtICJetAnalyzer;
    delete thePtAKJetAnalyzer;
  }

  if(theJPTJetAnalyzerFlag)        delete theJPTJetAnalyzer;
  if(thePFJetAnalyzerFlag)         delete thePFJetAnalyzer;

  if(theCaloMETAnalyzerFlag){
    delete theCaloMETAnalyzer;
    delete theCaloMETNoHFAnalyzer;
    delete theCaloMETHOAnalyzer;
    delete theCaloMETNoHFHOAnalyzer;
  }
  if(theTcMETAnalyzerFlag)         delete theTcMETAnalyzer;
  if(theMuCorrMETAnalyzerFlag)     delete theMuCorrMETAnalyzer;
  if(thePfMETAnalyzerFlag)         delete thePfMETAnalyzer;
  if(theHTMHTAnalyzerFlag)         delete theHTMHTAnalyzer;

}

// ***********************************************************
void JetMETAnalyzer::beginJob(void) {

  metname = "JetMETAnalyzer";

  LogTrace(metname)<<"[JetMETAnalyzer] Parameters initialization";
  dbe = edm::Service<DQMStore>().operator->();

  //
  //--- Jet
  if(theJetAnalyzerFlag) { 
    theSCJetAnalyzer->beginJob(dbe); 
    theAKJetAnalyzer->beginJob(dbe);
    if(theIConeJetAnalyzerFlag) theICJetAnalyzer->beginJob(dbe);
  }
  if(theJetCleaningFlag) {
    theCleanedSCJetAnalyzer->beginJob(dbe); 
    theCleanedAKJetAnalyzer->beginJob(dbe);
     if(theIConeJetAnalyzerFlag)theCleanedICJetAnalyzer->beginJob(dbe);
  }
  if(theJetPtAnalyzerFlag ) {
    thePtAKJetAnalyzer->beginJob(dbe);
    thePtSCJetAnalyzer->beginJob(dbe);
    thePtICJetAnalyzer->beginJob(dbe);
  }

  if(theJPTJetAnalyzerFlag) theJPTJetAnalyzer->beginJob(dbe);
  if(thePFJetAnalyzerFlag)  thePFJetAnalyzer->beginJob(dbe);

  //
  //--- MET
  if(theCaloMETAnalyzerFlag){
    theCaloMETAnalyzer->beginJob(dbe);
    theCaloMETNoHFAnalyzer->beginJob(dbe);
    theCaloMETHOAnalyzer->beginJob(dbe);
    theCaloMETNoHFHOAnalyzer->beginJob(dbe);
  }
  if(theTcMETAnalyzerFlag) theTcMETAnalyzer->beginJob(dbe);
  if(theMuCorrMETAnalyzerFlag) theMuCorrMETAnalyzer->beginJob(dbe);
  if(thePfMETAnalyzerFlag) thePfMETAnalyzer->beginJob(dbe);
  if(theHTMHTAnalyzerFlag) theHTMHTAnalyzer->beginJob(dbe);
  
  dbe->setCurrentFolder("JetMET");
  lumisecME = dbe->book1D("lumisec", "lumisec", 500, 0., 500.);

}

// ***********************************************************
void JetMETAnalyzer::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{
  //LogDebug("JetMETAnalyzer") << "beginRun, run " << run.id();

  //
  //--- htlConfig_
  //processname_="HLT";
  hltConfig_.init(processname_);
  if (!hltConfig_.init(processname_)) {
    processname_ = "FU";
    if (!hltConfig_.init(processname_)){
      LogDebug("JetMETAnalyzer") << "HLTConfigProvider failed to initialize.";
    }
  }

  if (hltConfig_.size()){
    dbe->setCurrentFolder("JetMET");
    hltpathME = dbe->book1D("hltpath", "hltpath", 300, 0., 300.);
  }

  for (unsigned int j=0; j!=hltConfig_.size(); ++j) {
    hltpathME->setBinLabel(j+1,hltConfig_.triggerName(j));
  }

  //
  //--- Jet

  //
  //--- MET
  if(theCaloMETAnalyzerFlag){
    theCaloMETAnalyzer->beginRun(iRun, iSetup);
    theCaloMETNoHFAnalyzer->beginRun(iRun, iSetup);
    theCaloMETHOAnalyzer->beginRun(iRun, iSetup);
    theCaloMETNoHFHOAnalyzer->beginRun(iRun, iSetup);
  }
  if(theTcMETAnalyzerFlag) theTcMETAnalyzer->beginRun(iRun, iSetup);
  if(theMuCorrMETAnalyzerFlag) theMuCorrMETAnalyzer->beginRun(iRun, iSetup);
  if(thePfMETAnalyzerFlag) thePfMETAnalyzer->beginRun(iRun, iSetup);
  //if(theHTMHTAnalyzerFlag) theHTMHTAnalyzer->beginRun(iRun, iSetup);

}

// ***********************************************************
void JetMETAnalyzer::endRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{

  //
  //--- Jet

  //
  //--- MET
  if(theCaloMETAnalyzerFlag){
    theCaloMETAnalyzer->endRun(iRun, iSetup, dbe);
    theCaloMETNoHFAnalyzer->endRun(iRun, iSetup, dbe);
    theCaloMETHOAnalyzer->endRun(iRun, iSetup, dbe);
    theCaloMETNoHFHOAnalyzer->endRun(iRun, iSetup, dbe);
  }
  if(theTcMETAnalyzerFlag) theTcMETAnalyzer->endRun(iRun, iSetup, dbe);
  if(theMuCorrMETAnalyzerFlag) theMuCorrMETAnalyzer->endRun(iRun, iSetup, dbe);
  if(thePfMETAnalyzerFlag) thePfMETAnalyzer->endRun(iRun, iSetup, dbe);
  //if(theHTMHTAnalyzerFlag) theHTMHTAnalyzer->endRun(iRun, iSetup, dbe);

}

// ***********************************************************
void JetMETAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {

  LogTrace(metname)<<"[JetMETAnalyzer] Analysis of event # ";

  // *** Fill lumisection ME
  int myLuminosityBlock;
  myLuminosityBlock = iEvent.luminosityBlock();
  lumisecME->Fill(myLuminosityBlock);

  if (myLuminosityBlock<_LSBegin) return;
  if (myLuminosityBlock>_LSEnd && _LSEnd>0) return;

  // **** Get the TriggerResults container
  edm::Handle<TriggerResults> triggerResults;
  iEvent.getByLabel(theTriggerResultsLabel, triggerResults);

  // *** Fill trigger results ME
  //if (&triggerResults){
  if (triggerResults.isValid()){
    edm::TriggerNames triggerNames;
    triggerNames.init(*(triggerResults.product()));
    for (unsigned int j=0; j!=hltConfig_.size(); ++j) {
      if (triggerResults->accept(j)){
        hltpathME->Fill(j);
      }
    }
  }

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

  // ==========================================================
  //Vertex information
  
  bool bPrimaryVertex = false;

  Handle<VertexCollection> vertexHandle;
  iEvent.getByLabel("offlinePrimaryVertices", vertexHandle);
  if ( vertexHandle.isValid() ){
    VertexCollection vertexCollection = *(vertexHandle.product());
    int vertex_number     = vertexCollection.size();
    VertexCollection::const_iterator v = vertexCollection.begin();
    double vertex_chi2    = v->normalizedChi2(); //v->chi2();
    //double vertex_d0      = sqrt(v->x()*v->x()+v->y()*v->y());
    double vertex_numTrks = v->tracksSize();
    double vertex_sumTrks = 0.0;
    //double vertex_Z       = v->z();
    for (Vertex::trackRef_iterator vertex_curTrack = v->tracks_begin(); vertex_curTrack!=v->tracks_end(); vertex_curTrack++) {
      vertex_sumTrks += (*vertex_curTrack)->pt();
    }
//     std::cout << v->x() << " " << v->y()<< " " << v->z() << " "
// 	      << v->chi2() << " " << v->tracksSize() << " "
// 	      << vertex_number << " " << v->normalizedChi2()
// 	      << std::endl;

    if (vertex_number>=1
        && vertex_numTrks>=2 
	&& vertex_chi2   <2.4 ) bPrimaryVertex = true;
    //&& fabs(vertex_Z)<20.0 ) bPrimaryVertex = true;

  }


  // **** Get the Calo Jet container
  edm::Handle<reco::CaloJetCollection> caloJets;
  
  // **** Get the AntiKt Jet container
  iEvent.getByLabel(theAKJetCollectionLabel, caloJets);    
  if(caloJets.isValid()){
    theAKJetAnalyzer->setJetHiPass(JetHiPass);
    theAKJetAnalyzer->setJetLoPass(JetLoPass);
    theAKJetAnalyzer->analyze(iEvent, iSetup, *caloJets);
  }

   if(caloJets.isValid() && bPrimaryVertex){
     theCleanedAKJetAnalyzer->setJetHiPass(JetHiPass);
     theCleanedAKJetAnalyzer->setJetLoPass(JetLoPass);
     theCleanedAKJetAnalyzer->analyze(iEvent, iSetup, *caloJets);
   }  
 
   if(caloJets.isValid()){
      if(theJetPtAnalyzerFlag){
        LogTrace(metname)<<"[JetMETAnalyzer] Call to the Jet Pt SisCone analyzer";
        thePtAKJetAnalyzer->analyze(iEvent, iSetup, *caloJets);
      }
   }


  // **** Get the SISCone Jet container
  iEvent.getByLabel(theSCJetCollectionLabel, caloJets);    
  if(caloJets.isValid()){
    theSCJetAnalyzer->setJetHiPass(JetHiPass);
    theSCJetAnalyzer->setJetLoPass(JetLoPass);
    theSCJetAnalyzer->analyze(iEvent, iSetup, *caloJets);
  }

   if(caloJets.isValid() && bPrimaryVertex){
     theCleanedSCJetAnalyzer->setJetHiPass(JetHiPass);
     theCleanedSCJetAnalyzer->setJetLoPass(JetLoPass);
     theCleanedSCJetAnalyzer->analyze(iEvent, iSetup, *caloJets);
   }

  if(caloJets.isValid()){
      if(theJetPtAnalyzerFlag){
        LogTrace(metname)<<"[JetMETAnalyzer] Call to the Jet Pt SisCone analyzer";
        thePtSCJetAnalyzer->analyze(iEvent, iSetup, *caloJets);
      }
  }

  // **** Get the Iterative Cone Jet container  
   iEvent.getByLabel(theICJetCollectionLabel, caloJets);
   if(theIConeJetAnalyzerFlag) {
   if(caloJets.isValid()){
    theICJetAnalyzer->setJetHiPass(JetHiPass);
    theICJetAnalyzer->setJetLoPass(JetLoPass);
    theICJetAnalyzer->analyze(iEvent, iSetup, *caloJets);	
   }

   if(caloJets.isValid() && bPrimaryVertex){
     theCleanedICJetAnalyzer->setJetHiPass(JetHiPass);
     theCleanedICJetAnalyzer->setJetLoPass(JetLoPass);
     theCleanedICJetAnalyzer->analyze(iEvent, iSetup, *caloJets);
   }
  }
   if(caloJets.isValid()){
      if(theJetPtAnalyzerFlag){
        LogTrace(metname)<<"[JetMETAnalyzer] Call to the Jet Pt ICone analyzer";
        thePtICJetAnalyzer->analyze(iEvent, iSetup, *caloJets);
      }
  }


// **** Get the JPT Jet container
   iEvent.getByLabel(theJPTJetCollectionLabel, caloJets);
   if(caloJets.isValid()){
     //theJPTJetAnalyzer->setJetHiPass(JetHiPass);
     //theJPTJetAnalyzer->setJetLoPass(JetLoPass);
     theJPTJetAnalyzer->analyze(iEvent, iSetup, *caloJets);
   }
   
  // **** Get the PFlow Jet container
  edm::Handle<reco::PFJetCollection> pfJets;
  iEvent.getByLabel(thePFJetCollectionLabel, pfJets);

  if(pfJets.isValid()){
    thePFJetAnalyzer->setJetHiPass(JetHiPass);
    thePFJetAnalyzer->setJetLoPass(JetLoPass);
    if(thePFJetAnalyzerFlag){
      LogTrace(metname)<<"[JetMETAnalyzer] Call to the PFJet analyzer";
      thePFJetAnalyzer->analyze(iEvent, iSetup, *pfJets);
    }
  } else {
    if (DEBUG) LogTrace(metname)<<"[JetMETAnalyzer] pfjets NOT VALID!!";
  }
  
  //
  // **** CaloMETAnalyzer **** //
  //
  if(theCaloMETAnalyzerFlag){

    theCaloMETAnalyzer->analyze(iEvent,       iSetup, *triggerResults);
    theCaloMETNoHFAnalyzer->analyze(iEvent,   iSetup, *triggerResults);
    theCaloMETHOAnalyzer->analyze(iEvent,     iSetup, *triggerResults);
    theCaloMETNoHFHOAnalyzer->analyze(iEvent, iSetup, *triggerResults);

  }

  //
  // **** TcMETAnalyzer **** //
  //
  if(theTcMETAnalyzerFlag){

    theTcMETAnalyzer->analyze(iEvent, iSetup, *triggerResults);

  }

  //
  // **** MuCorrMETAnalyzer **** //
  //
  if(theMuCorrMETAnalyzerFlag){

    theMuCorrMETAnalyzer->analyze(iEvent, iSetup, *triggerResults);

  }

  //
  // **** PfMETAnalyzer **** //
  //
  if(thePfMETAnalyzerFlag){
    
    thePfMETAnalyzer->analyze(iEvent, iSetup, *triggerResults);

  }

  //
  // **** HTMHTAnalyzer **** //
  //
  if(theHTMHTAnalyzerFlag){

    theHTMHTAnalyzer->analyze(iEvent, iSetup, *triggerResults);

  }

}

// ***********************************************************
void JetMETAnalyzer::endJob(void) {
  LogTrace(metname)<<"[JetMETAnalyzer] Saving the histos";
  bool outputMEsInRootFile   = parameters.getParameter<bool>("OutputMEsInRootFile");
  std::string outputFileName = parameters.getParameter<std::string>("OutputFileName");

  if(outputMEsInRootFile){
    //dbe->showDirStructure();
    dbe->save(outputFileName);
  }

  if(theCaloMETAnalyzerFlag){
    theCaloMETAnalyzer->endJob();
    theCaloMETNoHFAnalyzer->endJob();
    theCaloMETHOAnalyzer->endJob();
    theCaloMETNoHFHOAnalyzer->endJob();
  }
  if(theTcMETAnalyzerFlag) theTcMETAnalyzer->endJob();
  if(theMuCorrMETAnalyzerFlag) theMuCorrMETAnalyzer->endJob();
  if(thePfMETAnalyzerFlag)   thePfMETAnalyzer->endJob();
  //if(theHTMHTAnalyzerFlag) theHTMHTAnalyzer->endJob();

  if(theJPTJetAnalyzerFlag) theJPTJetAnalyzer->endJob();

}

