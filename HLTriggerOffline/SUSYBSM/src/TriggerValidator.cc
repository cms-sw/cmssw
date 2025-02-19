// -*- C++ -*-
//
// Package:    TriggerValidator
// Class:      TriggerValidator
// 
/**\class TriggerValidator TriggerValidator.cc HLTriggerOffline/SUSYBSM/src/TriggerValidator.cc

Description: Class to validate the Trigger Performance of the SUSYBSM group

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Massimiliano Chiorboli
//                   Maurizio Pierini
//                   Maria Spiropulu
//         Created:  Wed Aug 29 15:10:56 CEST 2007
//  Philip Hebda, July 2009
// $Id: TriggerValidator.cc,v 1.24 2010/12/14 17:20:35 vlimant Exp $
//
//


// system include files
#include <memory>
#include <stdio.h>
#include <iomanip>

#include "HLTriggerOffline/SUSYBSM/interface/TriggerValidator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Common/interface/Handle.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"

//Added by Max for the Trigger
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"
#include "DataFormats/Common/interface/TriggerResults.h"
//#include "DataFormats/L1Trigger/interface/L1ParticleMap.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"


//Added for the DQM
#include "DQMServices/Core/interface/MonitorElement.h"
#include "FWCore/Framework/interface/Run.h"

#include "TFile.h"
#include "TDirectory.h"
#include "TH1.h"
#include "TH2.h"
#include "TList.h"
#include "TObjString.h"
#include "TString.h"
#include "TObject.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//



//
// constructors and destructor
//

using namespace edm;
using namespace std;

TriggerValidator::TriggerValidator(const edm::ParameterSet& iConfig):
  dirname_(iConfig.getUntrackedParameter("dirname",
					      std::string("HLT/SusyExo"))),
  HistoFileName(iConfig.getUntrackedParameter("histoFileName",
					      std::string("SusyBsmTriggerValidation.root"))),
  StatFileName(iConfig.getUntrackedParameter("statFileName",
					      std::string("SusyBsmTriggerValidation.stat"))),
  l1Label(iConfig.getParameter<edm::InputTag>("L1Label")),
  hltLabel(iConfig.getParameter<edm::InputTag>("HltLabel")),
  mcFlag(iConfig.getUntrackedParameter<bool>("mc_flag",false)),
  l1Flag(iConfig.getUntrackedParameter<bool>("l1_flag",false)),
  reco_parametersets(iConfig.getParameter<VParameterSet>("reco_parametersets")),
  mc_parametersets(iConfig.getParameter<VParameterSet>("mc_parametersets")),
  turnOn_params(iConfig.getParameter<ParameterSet>("TurnOnParams")),
  plotMakerL1Input(iConfig.getParameter<ParameterSet>("PlotMakerL1Input")),
  plotMakerRecoInput(iConfig.getParameter<ParameterSet>("PlotMakerRecoInput")),
  muonTag_(iConfig.getParameter<edm::InputTag>("muonTag")),
  triggerTag_(iConfig.getParameter<edm::InputTag>("triggerTag")),
  processName_(iConfig.getParameter<std::string>("hltConfigName")),
  triggerName_(iConfig.getParameter<std::string>("triggerName"))
{
  //now do what ever initialization is needed
  theHistoFile = 0;
  nEvTot = 0;
  for(unsigned int i=0; i<reco_parametersets.size(); ++i) nEvRecoSelected.push_back(0);
  for(unsigned int i=0; i<mc_parametersets.size(); ++i) nEvMcSelected.push_back(0);

  nHltPaths = 0;
  nL1Bits = 0;


  // --- set the names in the dbe folders ---
  triggerBitsDir = "/TriggerBits";
  recoSelBitsDir = "/RecoSelection"; 
  mcSelBitsDir = "/McSelection";      


  LogDebug("TriggerValidator") << "constructor...." ;

  dbe_ = Service < DQMStore > ().operator->();
  if ( ! dbe_ ) {
    LogInfo("TriggerValidator") << "unabel to get DQMStore service?";
  }
  if (iConfig.getUntrackedParameter < bool > ("DQMStore", false)) {
    dbe_->setVerbose(0);
  }

  

  
  if (dbe_ != 0 ) {
    dbe_->setCurrentFolder(dirname_);
  }


   plotMakerL1Input.addParameter<std::string>("dirname",dirname_);
   plotMakerRecoInput.addParameter<std::string>("dirname",dirname_);

}


TriggerValidator::~TriggerValidator()
{
 
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
void
TriggerValidator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  using namespace edm;
  
  nEvTot++;
  
  vector<bool> eventRecoSelected, eventMcSelected;
  eventRecoSelected.resize(reco_parametersets.size());
  eventMcSelected.resize(mc_parametersets.size());
  for(unsigned int i=0; i<eventRecoSelected.size(); ++i) eventRecoSelected[i] =  myRecoSelector[i]->isSelected(iEvent);
  for(unsigned int i=0; i<eventMcSelected.size(); ++i) eventMcSelected[i] = mcFlag ? myMcSelector[i]->isSelected(iEvent) : false;

  for(unsigned int i=0; i<nEvRecoSelected.size(); ++i) if(eventRecoSelected[i]) nEvRecoSelected[i]++;
  for(unsigned int i=0; i<nEvMcSelected.size(); ++i) if(eventMcSelected[i]) nEvMcSelected[i]++;
  
  
  // ******************************************************** 
  // Get the L1 Info
  // ********************************************************    
  Handle<L1GlobalTriggerReadoutRecord> L1GTRR;
  //  try {iEvent.getByType(L1GTRR);} catch (...) {;}
  iEvent.getByLabel("gtDigis",L1GTRR);
  if (!L1GTRR.isValid()) {edm::LogWarning("Readout Error|L1") << "L1ParticleMapCollection Not Valid!";}
  int nL1size = L1GTRR->decisionWord().size();
  if(firstEvent) {

    //this piece of code concerns efficiencies
    //it must be moved to the client

//     //resize the eff and overlap vectors ccording to the number of L1 paths
//     effL1BeforeCuts.resize(L1GTRR->decisionWord().size()+1);
//     effL1AfterRecoCuts.resize(L1GTRR->decisionWord().size()+1);
//     effL1AfterMcCuts.resize(L1GTRR->decisionWord().size()+1);
     vCorrL1.resize(L1GTRR->decisionWord().size());
    for(unsigned int i=0; i<L1GTRR->decisionWord().size(); i++) {vCorrL1[i].resize(L1GTRR->decisionWord().size());}
    vCorrNormL1.resize(L1GTRR->decisionWord().size());
    for(unsigned int i=0; i<L1GTRR->decisionWord().size(); i++) {vCorrNormL1[i].resize(L1GTRR->decisionWord().size());}


    //Get the names of the L1 paths
    //for the moment the names are not included in L1GlobalTriggerReadoutRecord
    //we need to use L1GlobalTriggerObjectMapRecord
    edm::Handle<L1GlobalTriggerObjectMapRecord> gtObjectMapRecord;
    //    iEvent.getByLabel("l1GtEmulDigis", gtObjectMapRecord);
    //    iEvent.getByLabel("hltL1GtObjectMap", gtObjectMapRecord);
    iEvent.getByLabel(l1Label, gtObjectMapRecord);
     const std::vector<L1GlobalTriggerObjectMap>& objMapVec =
      gtObjectMapRecord->gtObjectMap();
     for (std::vector<L1GlobalTriggerObjectMap>::const_iterator itMap = objMapVec.begin();
	 itMap != objMapVec.end(); ++itMap) {
      int algoBit = (*itMap).algoBitNumber();
       std::string algoNameStr = (*itMap).algoName();
       l1NameMap[algoBit] = algoNameStr;
    }
    //resize the name vector and get the names
    l1Names_.resize(L1GTRR->decisionWord().size()+1);
    for (unsigned int i=0; i!=L1GTRR->decisionWord().size(); i++) {    
      l1Names_[i]=l1NameMap[i];
    }
    l1Names_[L1GTRR->decisionWord().size()] = "Total";
    
    //set the names of the bins for the "path" histos
    for(unsigned int i=0; i<l1Names_.size(); ++i) {
      hL1PathsBeforeCuts->setBinLabel(i+1,l1Names_[i].c_str(),1);
      for(unsigned int j=0; j<hL1PathsAfterRecoCuts.size(); ++j) hL1PathsAfterRecoCuts[j]->setBinLabel(i+1,l1Names_[i].c_str(),1);
      for(unsigned int j=0; j<hL1PathsAfterMcCuts.size(); ++j) hL1PathsAfterMcCuts[j]->setBinLabel(i+1,l1Names_[i].c_str(),1);
    }
  }
  
 //fill the eff vectors and histos for L1
  for (int i=0; i<nL1size; ++i) {
    l1bits.push_back(L1GTRR->decisionWord()[i]);
    if(L1GTRR->decisionWord()[i]) {
      numTotL1BitsBeforeCuts[i]++;
      hL1BitsBeforeCuts->Fill(i);
      hL1PathsBeforeCuts->Fill(i);
      for(unsigned int j=0; j<eventRecoSelected.size(); ++j)
	if(eventRecoSelected[j]) {
	  numTotL1BitsAfterRecoCuts[j][i]++;
	  hL1BitsAfterRecoCuts[j]->Fill(i);
	  hL1PathsAfterRecoCuts[j]->Fill(i);
	}
      for(unsigned int j=0; j<eventMcSelected.size(); ++j)
	if(eventMcSelected[j]) {
	  numTotL1BitsAfterMcCuts[j][i]++;
	  hL1BitsAfterMcCuts[j]->Fill(i);
	  hL1PathsAfterMcCuts[j]->Fill(i);
	}
    }      
  }

  //Calculate the overlap among l1 bits
  for(int i=0; i<nL1size; ++i) {
    for(int j=0; j<nL1size; ++j) {
      if(l1bits[i]*l1bits[j]) vCorrL1[i][j]++;
    }
  }

  //fill the last bin with the total of events
  numTotL1BitsBeforeCuts[nL1size]++;
  hL1BitsBeforeCuts->Fill(nL1size);
  hL1PathsBeforeCuts->Fill(nL1size);
  for(unsigned int i=0; i<eventRecoSelected.size(); ++i)
    if(eventRecoSelected[i]) {
      numTotL1BitsAfterRecoCuts[i][nL1size]++;
      hL1BitsAfterRecoCuts[i]->Fill(nL1size);
      hL1PathsAfterRecoCuts[i]->Fill(nL1size);
    }
  for(unsigned int i=0; i<eventMcSelected.size(); ++i)
    if(eventMcSelected[i]) {
      numTotL1BitsAfterMcCuts[i][nL1size]++;
      hL1BitsAfterMcCuts[i]->Fill(nL1size);
      hL1PathsAfterMcCuts[i]->Fill(nL1size);
    }


  // ******************************************************** 
  // Get the HLT Info
  // ******************************************************** 
  edm::Handle<TriggerResults> trhv;
  iEvent.getByLabel(hltLabel,trhv);
  if( ! trhv.isValid() ) {
    LogDebug("") << "HL TriggerResults with label ["+hltLabel.encode()+"] not found!";
    return;
  }  


//   if(!trhv.isValid()) {
//     std::cout << "invalid handle for HLT TriggerResults" << std::endl;
//   } 


  if(firstEvent) {


 
    //
    // The following piece of code concerns efficiencies and so must be moved to the client
    //


//     //resize the eff and overlap vectors ccording to the number of L1 paths
//     effHltBeforeCuts.resize(trhv->size()+1);
//     effHltAfterRecoCuts.resize(trhv->size()+1);
//     effHltAfterMcCuts.resize(trhv->size()+1);
     vCorrHlt.resize(trhv->size());
    for(unsigned int i=0; i<trhv->size(); i++) {vCorrHlt[i].resize(trhv->size());}
    vCorrNormHlt.resize(trhv->size());
    for(unsigned int i=0; i<trhv->size(); i++) {vCorrNormHlt[i].resize(trhv->size());}


    //resize the name vector and get the names
    const edm::TriggerNames & triggerNames = iEvent.triggerNames(*trhv);
    hlNames_=triggerNames.triggerNames();
    hlNames_.push_back("Total");

    //set the bin names for the "path" histos
    for (unsigned int i=0; i<hlNames_.size(); ++i) {
      hHltPathsBeforeCuts->setBinLabel(i+1,hlNames_[i].c_str(),1);
      for(unsigned int j=0; j<hHltPathsAfterRecoCuts.size(); ++j) hHltPathsAfterRecoCuts[j]->setBinLabel(i+1,hlNames_[i].c_str(),1);
      for(unsigned int j=0; j<hHltPathsAfterMcCuts.size(); ++j) hHltPathsAfterMcCuts[j]->setBinLabel(i+1,hlNames_[i].c_str(),1);
    }
  }

  //fill the eff vectors and histos for HLT
  for(unsigned int i=0; i< trhv->size(); i++) {
    hltbits.push_back(trhv->at(i).accept());
    if(trhv->at(i).accept()) {
      numTotHltBitsBeforeCuts[i]++;
      hHltBitsBeforeCuts->Fill(i);
      hHltPathsBeforeCuts->Fill(i);
      for(unsigned int j=0; j<eventRecoSelected.size(); ++j)
	if(eventRecoSelected[j]) {
	  numTotHltBitsAfterRecoCuts[j][i]++;
	  hHltBitsAfterRecoCuts[j]->Fill(i);
	  hHltPathsAfterRecoCuts[j]->Fill(i);
	}
      for(unsigned int j=0; j<eventMcSelected.size(); ++j)
	if(eventMcSelected[j]) {
	  numTotHltBitsAfterMcCuts[j][i]++;
	  hHltBitsAfterMcCuts[j]->Fill(i);
	  hHltPathsAfterMcCuts[j]->Fill(i);
	}
    }      
  }

  //Calculate the overlap among HLT paths
 for(unsigned int i=0; i< trhv->size(); i++) {
   for(unsigned int j=0; j< trhv->size(); j++) {
//      cout << "trhv->size() = " << trhv->size() << endl;
//      cout << "hltbits["<< i << "] = " << hltbits[i] << endl;
//      cout << "hltbits["<< j << "] = " << hltbits[j] << endl;
     if(hltbits[i]*hltbits[j]) vCorrHlt[i][j]++;
   }
 }


  //The overlap histos are filled in the endJob() method

  //fill the last bin with the total of events
  numTotHltBitsBeforeCuts[trhv->size()]++;
  hHltBitsBeforeCuts->Fill(trhv->size());
  hHltPathsBeforeCuts->Fill(trhv->size());
  for(unsigned int i=0; i<eventRecoSelected.size(); ++i)
    if(eventRecoSelected[i]) {
      numTotHltBitsAfterRecoCuts[i][trhv->size()]++;
      hHltBitsAfterRecoCuts[i]->Fill(trhv->size());
      hHltPathsAfterRecoCuts[i]->Fill(trhv->size());
    }
  for(unsigned int i=0; i<eventMcSelected.size(); ++i)
    if(eventMcSelected[i]) {
      numTotHltBitsAfterMcCuts[i][trhv->size()]++;
      hHltBitsAfterMcCuts[i]->Fill(trhv->size());
      hHltPathsAfterMcCuts[i]->Fill(trhv->size());
    }



  if(firstEvent) {
    if(l1Flag) myPlotMakerL1->bookHistos(dbe_,&l1bits,&hltbits,&l1Names_,&hlNames_);
    myPlotMakerReco->bookHistos(dbe_,&l1bits,&hltbits,&l1Names_,&hlNames_);
    //    myTurnOnMaker->bookHistos();
  }
  if(l1Flag) myPlotMakerL1->fillPlots(iEvent);
  myPlotMakerReco->fillPlots(iEvent);
  //  myTurnOnMaker->fillPlots(iEvent);

  firstEvent = false;

  myMuonAnalyzer->FillPlots(iEvent, iSetup);
  l1bits.clear();
  hltbits.clear();
}


void TriggerValidator::beginJob(){
}



// ------------ method called once each job just before starting event loop  ------------
void TriggerValidator::beginRun(const edm::Run& run, const edm::EventSetup& c)
{


  DQMStore *dbe_ = 0;
  dbe_ = Service<DQMStore>().operator->();
  
  if (dbe_) {
    dbe_->setCurrentFolder(dirname_);
    dbe_->rmdir(dirname_);
  }
  
  if (dbe_) {
    dbe_->setCurrentFolder(dirname_);
  }  
  
  
  bool changed(true);
  //  cout << "hltConfig_.init(run,c,processName_,changed) = " << (int) hltConfig_.init(run,c,processName_,changed) << endl;
  //  cout << "changed = " << (int) changed << endl;
  if (hltConfig_.init(run,c,processName_,changed)) {
    //    cout << "AAAA" << endl;
    if (changed) {
      //     cout << "BBBBBBB" << endl;
     // check if trigger name in (new) config
      if (triggerName_!="@") { // "@" means: analyze all triggers in config
	//	cout << "hltConfig_.size() = " << hltConfig_.size() << endl;
	nHltPaths = hltConfig_.size();
	const unsigned int triggerIndex(hltConfig_.triggerIndex(triggerName_));
	if (triggerIndex>=nHltPaths) {
// 	  cout << "HLTriggerOffline/SUSYBSM"
// 	       << " TriggerName " << triggerName_ 
// 	       << " not available in (new) config!" << endl;
// 	  cout << "Available TriggerNames are: " << endl;
	  hltConfig_.dump("Triggers");
	}
      }
      else {
	//     cout << "CCCCCCCC" << endl;
	nHltPaths = hltConfig_.size();
      }
    }
  } else {
//     cout << "HLTriggerOffline/SUSYBSM"
// 	 << " config extraction failure with process name "
// 	 << processName_ << endl;
  }

  //  cout << "nHltPaths = " << nHltPaths << endl;
  nL1Bits = 128; 

  

  for(unsigned int i=0; i<reco_parametersets.size(); ++i) myRecoSelector.push_back(new RecoSelector(reco_parametersets[i]));
  if(mcFlag) for(unsigned int i=0; i<mc_parametersets.size(); ++i) myMcSelector.push_back(new McSelector(mc_parametersets[i]));
  if(l1Flag) myPlotMakerL1     = new PlotMakerL1(plotMakerL1Input);
  myPlotMakerReco   = new PlotMakerReco(plotMakerRecoInput);
//   myTurnOnMaker = new TurnOnMaker(turnOn_params);
  firstEvent = true;

  //resize the vectors ccording to the number of L1 paths
  numTotL1BitsBeforeCuts.resize(nL1Bits+1);
  numTotL1BitsAfterRecoCuts.resize(reco_parametersets.size());
  for(unsigned int i=0; i<numTotL1BitsAfterRecoCuts.size(); ++i) numTotL1BitsAfterRecoCuts[i].resize(nL1Bits+1);
  numTotL1BitsAfterMcCuts.resize(mc_parametersets.size());
  for(unsigned int i=0; i<numTotL1BitsAfterMcCuts.size(); ++i) numTotL1BitsAfterMcCuts[i].resize(nL1Bits+1);

  //resize the vectors ccording to the number of HLT paths
  numTotHltBitsBeforeCuts.resize(nHltPaths+1);
  numTotHltBitsAfterRecoCuts.resize(reco_parametersets.size());
  for(unsigned int i=0; i<numTotHltBitsAfterRecoCuts.size(); ++i) numTotHltBitsAfterRecoCuts[i].resize(nHltPaths+1);
  numTotHltBitsAfterMcCuts.resize(mc_parametersets.size());
  for(unsigned int i=0; i<numTotHltBitsAfterMcCuts.size(); ++i) numTotHltBitsAfterMcCuts[i].resize(nHltPaths+1);
  
  if (dbe_) {
    dbe_->setCurrentFolder(dirname_);
    dbe_->rmdir(dirname_);
  }
  
  
  if (dbe_) {
    dbe_->setCurrentFolder(dirname_);
    }
  
  dbe_->setCurrentFolder(dirname_+triggerBitsDir);
  //add 1 bin for the Total
  hL1BitsBeforeCuts  = dbe_->book1D("L1Bits", "L1 Trigger Bits",nL1Bits+1, 0, nL1Bits+1);    
  hHltBitsBeforeCuts = dbe_->book1D("HltBits","HL Trigger Bits",nHltPaths+1, 0, nHltPaths+1);
//   hL1OverlapNormToTotal        = dbe_->book2D("L1OverlapNormToTotal"       ,"Overlap among L1 paths, norm to the Total number of Events", 1, 0, 1, 1, 0, 1);
//   hHltOverlapNormToTotal       = dbe_->book2D("HltOverlapNormToTotal"      ,"Overlap among Hlt paths, norm to the Total number of Events ", 1, 0, 1, 1, 0, 1);
//   hL1OverlapNormToLargestPath  = dbe_->book2D("L1OverlapNormToLargestPath" ,"Overlap among L1 paths, norm to the Largest of the couple ", 1, 0, 1, 1, 0, 1);
//   hHltOverlapNormToLargestPath = dbe_->book2D("HltOverlapNormToLargestPath","Overlap among Hlt paths, norm to the Largest of the couple ", 1, 0, 1, 1, 0, 1);

  for(unsigned int i=0; i<myRecoSelector.size(); ++i)
    {
      string path_name = myRecoSelector[i]->GetName();
      char histo_name[256], histo_title[256];
      //sprintf(histo_name, "L1Bits");
      sprintf(histo_name, "L1Bits_%s", path_name.c_str());
      sprintf(histo_title, "L1 Trigger Bits for %s Selection", path_name.c_str());
      dbe_->setCurrentFolder(dirname_+recoSelBitsDir+"/"+path_name);   
      hL1BitsAfterRecoCuts.push_back(dbe_->book1D(histo_name, histo_title, nL1Bits+1, 0, nL1Bits+1));   
      //sprintf(histo_name, "HltBits");
      sprintf(histo_name, "HltBits_%s", path_name.c_str());
      sprintf(histo_title, "HL Trigger Bits for %s Selection", path_name.c_str()); 
      hHltBitsAfterRecoCuts.push_back(dbe_->book1D(histo_name, histo_title, nHltPaths+1, 0, nHltPaths+1));
    }
  for(unsigned int i=0; i<myMcSelector.size(); ++i)
    {
      string path_name = myMcSelector[i]->GetName();
      char histo_name[256], histo_title[256];
      //sprintf(histo_name, "L1Bits");
      sprintf(histo_name, "L1Bits_%s", path_name.c_str());
      sprintf(histo_title, "L1 Trigger Bits for %s Selection", path_name.c_str());
      dbe_->setCurrentFolder(dirname_+mcSelBitsDir+"/"+path_name);   
      hL1BitsAfterMcCuts.push_back(dbe_->book1D(histo_name, histo_title, nL1Bits+1, 0, nL1Bits+1));
      //sprintf(histo_name, "HltBits");
      sprintf(histo_name, "HltBits_%s", path_name.c_str());
      sprintf(histo_title, "HL Trigger Bits for %s Selection", path_name.c_str()); 
      hHltBitsAfterMcCuts.push_back(dbe_->book1D(histo_name, histo_title, nHltPaths+1, 0, nHltPaths+1));
    }

  //create the histos with paths
  //identical to the ones with "bits"
  //but with the names in the x axis
  //instead of the numbers

  dbe_->setCurrentFolder(dirname_+triggerBitsDir);
  TH1F* hTemp = (TH1F*) (hL1BitsBeforeCuts->getTH1F())->Clone("L1Paths");
  hL1PathsBeforeCuts  = dbe_->book1D("L1Paths", hTemp);
  //  hL1PathsBeforeCuts  = dbe_->book1D("L1Paths", hL1BitsBeforeCuts->getTH1F());
  hTemp = (TH1F*) (hHltBitsBeforeCuts->getTH1F())->Clone("HltPaths");
  hHltPathsBeforeCuts = dbe_->book1D("HltPaths", hTemp);

  for(unsigned int i=0; i<myRecoSelector.size(); ++i)
    {
      string path_name = myRecoSelector[i]->GetName();
      char histo_name[256];
      //sprintf(histo_name, "L1Paths");
      sprintf(histo_name, "L1Paths_%s", path_name.c_str());
      dbe_->setCurrentFolder(dirname_+recoSelBitsDir+"/"+path_name);
      hTemp = (TH1F*) (hL1BitsAfterRecoCuts[i]->getTH1F())->Clone(histo_name);
      hL1PathsAfterRecoCuts.push_back(dbe_->book1D(histo_name, hTemp));
      //sprintf(histo_name, "HltPaths");
      sprintf(histo_name, "HltPaths_%s", path_name.c_str());
      hTemp = (TH1F*) (hHltBitsAfterRecoCuts[i]->getTH1F())->Clone(histo_name);
      hHltPathsAfterRecoCuts.push_back(dbe_->book1D(histo_name, hTemp));
    }

  for(unsigned int i=0; i<myMcSelector.size(); ++i)
    {
      string path_name = myMcSelector[i]->GetName();
      char histo_name[256];
      //sprintf(histo_name, "L1Paths");
      sprintf(histo_name, "L1Paths_%s", path_name.c_str());
      dbe_->setCurrentFolder(dirname_+mcSelBitsDir+"/"+path_name);
      hTemp = (TH1F*) (hL1BitsAfterMcCuts[i]->getTH1F())->Clone(histo_name);
      hL1PathsAfterMcCuts.push_back(dbe_->book1D(histo_name, hTemp));
      //sprintf(histo_name, "HltPaths");
      sprintf(histo_name, "HltPaths_%s", path_name.c_str());
      hTemp = (TH1F*) (hHltBitsAfterMcCuts[i]->getTH1F())->Clone(histo_name);
      hHltPathsAfterMcCuts.push_back(dbe_->book1D(histo_name, hTemp));
    }

  myMuonAnalyzer = new MuonAnalyzerSBSM(triggerTag_, muonTag_);
  myMuonAnalyzer->InitializePlots(dbe_, dirname_);
}




// ------------ method called once each job just after ending the event loop  ------------
void 
TriggerValidator::endRun(const edm::Run& run, const edm::EventSetup& c)
{

  //  myTurnOnMaker->finalOperations();

  //This piece of code concerns efficiencies
  //it must be moved to the client
  


//   //calculate the final efficiencies and the normalizations
//   for(unsigned int i=0; i<numTotL1BitsBeforeCuts.size()-1; i++) {
//     effL1BeforeCuts[i] = (double)numTotL1BitsBeforeCuts[i]/(double)nEvTot;
//     for(unsigned int j=0; j<numTotL1BitsBeforeCuts.size()-1; j++) {
//       vCorrNormL1[i][j] = (double)vCorrL1[i][j]/(double)nEvTot;
//     }
//   }

//   for(unsigned int i=0; i<numTotHltBitsBeforeCuts.size()-1; i++) {
//     effHltBeforeCuts[i] = (double)numTotHltBitsBeforeCuts[i]/(double)nEvTot;
//     for(unsigned int j=0; j<numTotHltBitsBeforeCuts.size()-1; j++) {
//       vCorrNormHlt[i][j] = (double)vCorrHlt[i][j]/(double)nEvTot;
//     }
//   }

//   //after the reco cuts

//   if(nEvRecoSelected) {
//     for(unsigned int i=0; i<numTotL1BitsAfterRecoCuts.size()-1; i++) {
//       effL1AfterRecoCuts[i] = (double)numTotL1BitsAfterRecoCuts[i]/(double)nEvRecoSelected;
//     }
    
//     for(unsigned int i=0; i<numTotHltBitsAfterRecoCuts.size()-1; i++) {
//       effHltAfterRecoCuts[i] = (double)numTotHltBitsAfterRecoCuts[i]/(double)nEvRecoSelected;
//     }
//   } else {
    
//     for(unsigned int i=0; i<numTotL1BitsAfterRecoCuts.size()-1; i++) {
//       effL1AfterRecoCuts[i] = -1;
//     }
    
//     for(unsigned int i=0; i<numTotHltBitsAfterRecoCuts.size()-1; i++) {
//       effHltAfterRecoCuts[i] = -1;
//     }
//   }


//   //after the mc cuts
//   if(nEvMcSelected) {
//     for(unsigned int i=0; i<numTotL1BitsAfterMcCuts.size()-1; i++) {
//       effL1AfterMcCuts[i] = (double)numTotL1BitsAfterMcCuts[i]/(double)nEvMcSelected;
//     }
    
//     for(unsigned int i=0; i<numTotHltBitsAfterMcCuts.size()-1; i++) {
//       effHltAfterMcCuts[i] = (double)numTotHltBitsAfterMcCuts[i]/(double)nEvMcSelected;
//     }
//   } else {
//     for(unsigned int i=0; i<numTotL1BitsAfterMcCuts.size()-1; i++) {
//       effL1AfterMcCuts[i] = -1;
//     }
    
//     for(unsigned int i=0; i<numTotHltBitsAfterMcCuts.size()-1; i++) {
//       effHltAfterMcCuts[i] = -1;
//     }
//   }    








//   myPlotMaker->writeHistos();
//   myTurnOnMaker->writeHistos();


  //  using namespace std;

  unsigned int n(l1Names_.size());

  n = l1Names_.size();
  edm::LogInfo("L1TableSummary") << endl;
  edm::LogVerbatim("L1TableSummary") << "L1T-Table "
       << right << setw(10) << "L1T  Bit#" << " "
       << "Name" << "\n";
  for (unsigned int i=0; i!=n; i++) {
    edm::LogVerbatim("L1TableSummary") << right << setw(20) << i << " "
	 << l1Names_[i] << "\n";
  }
  
  
  n = hlNames_.size();
  edm::LogInfo("HltTableSummary") << endl;
  edm::LogVerbatim("HltTableSummary") << "HLT-Table "
       << right << setw(10) << "HLT  Bit#" << " "
       << "Name" << "\n";
  
  for (unsigned int i=0; i!=n; i++) {
     edm::LogVerbatim("HltTableSummary") << right << setw(20) << i << " "
	 << hlNames_[i] << "\n";
  }
  
  edm::LogVerbatim("HltTableSummary") << endl;
  edm::LogVerbatim("HltTableSummary") << "HLT-Table end!" << endl;
  edm::LogVerbatim("HltTableSummary") << endl;
  

 
  //the stat file with the efficiecies has to be moved to the client



//  //Print in a stat file the efficiencies and the overlaps
 
 
//   ofstream statFile(StatFileName.c_str(),ios::out);


//   statFile << "*********************************************************************************" << endl;
//   statFile << "*********************************************************************************" << endl;
//   statFile << "                                   L1 Efficiencies                               " << endl;
//   statFile << "*********************************************************************************" << endl;
//   statFile << "*********************************************************************************" << endl;
//   statFile << endl;
//   statFile << "---------------------------------------------------------------------------------" << endl;
//   statFile << "---------------------------------------------------------------------------------" << endl;
//   statFile << "|           L1 Path             |   eff (Tot)    | eff (Reco Sel)|  eff (Mc Sel) |" << endl;
//   statFile << "---------------------------------------------------------------------------------" << endl;
//   statFile << "---------------------------------------------------------------------------------" << endl;
//   for(unsigned int i=0; i<numTotL1BitsBeforeCuts.size()-1; i++) {
//     statFile << "|  " << left << setw(29) << l1Names_[i] << "|" << setprecision(3) << showpoint << right << setw(13) << effL1BeforeCuts[i]    << "  |" <<
//                                                                                                             setw(13) << effL1AfterRecoCuts[i] << "  |" <<
//                                                                                                             setw(13) << effL1AfterMcCuts[i]   << "  |" << endl;
//   }
//   statFile << "---------------------------------------------------------------------------------" << endl;
//   statFile << "---------------------------------------------------------------------------------" << endl;
//   statFile << endl;
//   statFile << endl;
//   statFile << endl;
//   statFile << endl;
//   statFile << endl;
//   statFile << endl;



//   statFile << "**********************************************************************************" << endl;
//   statFile << "**********************************************************************************" << endl;
//   statFile << "                                  Hlt Efficiencies                                " << endl;
//   statFile << "**********************************************************************************" << endl;
//   statFile << "**********************************************************************************" << endl;
//   statFile << endl;
//   statFile << "----------------------------------------------------------------------------------" << endl;
//   statFile << "----------------------------------------------------------------------------------" << endl;
//   statFile << "|           Hlt Path             |   eff (Tot)    | eff (Reco Sel)|  eff (Mc Sel) |" << endl;
//   statFile << "----------------------------------------------------------------------------------" << endl;
//   statFile << "----------------------------------------------------------------------------------" << endl;
//   for(unsigned int i=0; i<numTotHltBitsBeforeCuts.size()-1; i++) {
//     statFile << "|  " << left << setw(29) << hlNames_[i] << "|" << setprecision(3) << showpoint << right << setw(13) << effHltBeforeCuts[i]    << "  |" << 
//                                                                                                             setw(13) << effHltAfterRecoCuts[i] << "  |" << 
//                                                                                                             setw(13) << effHltAfterMcCuts[i]   << "  |" <<endl;
//   }
//   statFile << "----------------------------------------------------------------------------------" << endl;
//   statFile << "----------------------------------------------------------------------------------" << endl;
//   statFile << endl;
//   statFile << endl;
//   statFile << endl;
//   statFile << endl;
//   statFile << endl;
//   statFile << endl;





//   statFile << "****************************************************************************************************************************************************" << endl; 
//   statFile << "****************************************************************************************************************************************************" << endl; 
//   statFile << "                                                      L1 Correlations   (only overlaps >5% are shown, and only without any selection)                                               " << endl;
//   statFile << "****************************************************************************************************************************************************" << endl; 
//   statFile << "****************************************************************************************************************************************************" << endl;
//   statFile << endl;
//   statFile << "----------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
//   statFile << "----------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
//   statFile << "|           L1 Path 1           |           L1 Path 2           |  Overlap Norm to Total  |  Overlap Norm to Path  |         Path of Norm          |" << endl;
//   statFile << "----------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
//   statFile << "----------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
//   statFile << endl;
//   for(unsigned int i=0; i<numTotL1BitsBeforeCuts.size()-1; i++) {
//     for(unsigned int j=0; j<numTotL1BitsBeforeCuts.size()-1; j++) {
//       if(vCorrNormL1[i][j]>0.05) {
// 	int iNorm = 0;
// 	if(effL1BeforeCuts[i] > effL1BeforeCuts[j]) {iNorm  = i;}
// 	else {iNorm = j;}
// 	double effNorm  =  vCorrNormL1[i][j] / effL1BeforeCuts[iNorm];
// 	statFile << "|  " << left << setw(29) << l1Names_[i] << "|  " << setw(29) <<  left << l1Names_[j] << "|"
// 		 << setprecision(3) << showpoint << right  << setw(22) << vCorrNormL1[i][j] << "   |"
// 		 << setprecision(3) << showpoint << right  << setw(21) << effNorm           << "   |  "
// 		 << left << setw(29) << l1Names_[iNorm] << "|" << endl;
//       }
//     }
//   statFile << "----------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
//   }
//   statFile << "----------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
//   statFile << "----------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
//   statFile << endl;
//   statFile << endl;
//   statFile << endl;
//   statFile << endl;
//   statFile << endl;
//   statFile << endl;


//   statFile << "****************************************************************************************************************************************************" << endl; 
//   statFile << "****************************************************************************************************************************************************" << endl; 
//   statFile << "                                                     Hlt Correlations   (only overlaps >5% are shown, and only without any selection)                                               " << endl;
//   statFile << "****************************************************************************************************************************************************" << endl; 
//   statFile << "****************************************************************************************************************************************************" << endl;
//   statFile << endl;
//   statFile << "----------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
//   statFile << "----------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
//   statFile << "|           Hlt Path 1          |           Hlt Path 2          |  Overlap Norm to Total  |  Overlap Norm to Path  |         Path of Norm          |" << endl;
//   statFile << "----------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
//   statFile << "----------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
//   statFile << endl;
//   for(unsigned int i=0; i<numTotHltBitsBeforeCuts.size()-1; i++) {
//     for(unsigned int j=0; j<numTotHltBitsBeforeCuts.size()-1; j++) {
//       if(vCorrNormHlt[i][j]>0.05) {
// 	int iNorm = 0;
// 	if(effHltBeforeCuts[i] > effHltBeforeCuts[j]) {iNorm  = i;}
// 	else {iNorm = j;}
// 	double effNorm  = vCorrNormHlt[i][j]/effHltBeforeCuts[iNorm];
// 	statFile << "|  " << left << setw(29) << hlNames_[i] << "|  " << setw(29) <<  left << hlNames_[j] << "|"
// 		 << setprecision(3) << showpoint << right  << setw(22) << vCorrNormHlt[i][j] << "   |"
// 		 << setprecision(3) << showpoint << right  << setw(21) << effNorm            << "   |  "
// 		 << left << setw(29) << hlNames_[iNorm] << "|" << endl;
//       }
//     }
//   statFile << "----------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
//   }
//   statFile << "----------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
//   statFile << "----------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
//   statFile << endl;
//   statFile << endl;
//   statFile << endl;
//   statFile << endl;
//   statFile << endl;
//   statFile << endl;




//   statFile.close();


  for(unsigned int i=0; i<myRecoSelector.size(); ++i) delete myRecoSelector[i];
  myRecoSelector.clear();
  if(mcFlag) 
    {
      for(unsigned int i=0; i<myMcSelector.size(); ++i) delete myMcSelector[i];
      myMcSelector.clear();
    }
     
  if(l1Flag) delete myPlotMakerL1;
  delete myPlotMakerReco;
//   delete myTurnOnMaker;
  return;
}


// - method called once each job just after ending the event loop  ------------
void 
TriggerValidator::endJob() 
{
  LogInfo("TriggerValidator") << "endJob";
   return;
}






//define this as a plug-in
DEFINE_FWK_MODULE(TriggerValidator);
