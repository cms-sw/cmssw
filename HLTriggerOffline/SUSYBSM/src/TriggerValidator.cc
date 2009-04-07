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
// $Id: TriggerValidator.cc,v 1.10 2009/01/29 18:40:31 chiorbo Exp $
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
#include "FWCore/Framework/interface/TriggerNames.h"
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
  userCut_params(iConfig.getParameter<ParameterSet>("UserCutParams")),
  turnOn_params(iConfig.getParameter<ParameterSet>("TurnOnParams")),
  objectList(iConfig.getParameter<ParameterSet>("ObjectList"))
{
  //now do what ever initialization is needed
  theHistoFile = 0;
  nEvTot = 0;
  nEvRecoSelected = 0;
  nEvMcSelected = 0;

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


  objectList.addParameter<std::string>("dirname",dirname_);

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
  
  bool eventRecoSelected = myRecoSelector->isSelected(iEvent);
  bool eventMcSelected   = mcFlag ? myMcSelector->isSelected(iEvent) : false;
  
  if(eventRecoSelected) nEvRecoSelected++;
  if(eventMcSelected) nEvMcSelected++;
  
  
  // ******************************************************** 
  // Get the L1 Info
  // ********************************************************    
  Handle<L1GlobalTriggerReadoutRecord> L1GTRR;
  //  try {iEvent.getByType(L1GTRR);} catch (...) {;}
  iEvent.getByLabel("gtDigis",L1GTRR);
  std::vector<int> l1bits;
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
    
  }
  
 //fill the eff vectors and histos for L1
  for (int i=0; i<nL1size; ++i) {
    l1bits.push_back(L1GTRR->decisionWord()[i]);
    if(L1GTRR->decisionWord()[i]) {
      numTotL1BitsBeforeCuts[i]++;
      hL1BitsBeforeCuts->Fill(i);
      if(eventRecoSelected) {
	numTotL1BitsAfterRecoCuts[i]++;
	hL1BitsAfterRecoCuts->Fill(i);
      }
      if(eventMcSelected) {
	numTotL1BitsAfterMcCuts[i]++;
	hL1BitsAfterMcCuts->Fill(i);
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
  if(eventRecoSelected) {
    numTotL1BitsAfterRecoCuts[nL1size]++;
   hL1BitsAfterRecoCuts->Fill(nL1size);
  }
  if(eventMcSelected) {
    numTotL1BitsAfterMcCuts[nL1size]++;
    hL1BitsAfterMcCuts->Fill(nL1size);
  }


  // ******************************************************** 
  // Get the HLT Info
  // ******************************************************** 
  edm::Handle<TriggerResults> trhv;
  iEvent.getByLabel(hltLabel,trhv);
  std::vector<int> hltbits;

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
    triggerNames_.init(*trhv);
    hlNames_=triggerNames_.triggerNames();
    hlNames_.push_back("Total");
  }

  //fill the eff vectors and histos for HLT
  for(unsigned int i=0; i< trhv->size(); i++) {
    hltbits.push_back(trhv->at(i).accept());
    if(trhv->at(i).accept()) {
      numTotHltBitsBeforeCuts[i]++;
      hHltBitsBeforeCuts->Fill(i);
      if(eventRecoSelected) {
	numTotHltBitsAfterRecoCuts[i]++;
	hHltBitsAfterRecoCuts->Fill(i);
      }
      if(eventMcSelected) {
	numTotHltBitsAfterMcCuts[i]++;
	hHltBitsAfterMcCuts->Fill(i);
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
  if(eventRecoSelected) {
    numTotHltBitsAfterRecoCuts[trhv->size()]++;
    hHltBitsAfterRecoCuts->Fill(trhv->size());
  }
  if(eventMcSelected) {
    numTotHltBitsAfterMcCuts[trhv->size()]++;
    hHltBitsAfterMcCuts->Fill(trhv->size());
  }



  if(firstEvent) {
    myPlotMaker->bookHistos(dbe_,&l1bits,&hltbits,&l1Names_,&hlNames_);
    //    myTurnOnMaker->bookHistos();
  }
  myPlotMaker->fillPlots(iEvent);
  //  myTurnOnMaker->fillPlots(iEvent);

  firstEvent = false;

}


// ------------ method called once each job just before starting event loop  ------------
void 
TriggerValidator::beginJob(const edm::EventSetup&)
{

  DQMStore *dbe = 0;
  dbe = Service<DQMStore>().operator->();
  
  
  if (dbe) {
    dbe->setCurrentFolder(dirname_);
    dbe->rmdir(dirname_);
  }
  
  
  if (dbe) {
    dbe->setCurrentFolder(dirname_);
  }  
  
  
  if (hltConfig_.init("HLT")) {
    nHltPaths = hltConfig_.size(); 
  }
  nL1Bits = 128; 

  


  myRecoSelector = new RecoSelector(userCut_params);
  if(mcFlag) myMcSelector   = new McSelector(userCut_params);
  myPlotMaker   = new PlotMaker(objectList);
//   myTurnOnMaker = new TurnOnMaker(turnOn_params);
  firstEvent = true;

  
  //resize the vectors ccording to the number of L1 paths
  numTotL1BitsBeforeCuts.resize(nL1Bits+1);
  numTotL1BitsAfterRecoCuts.resize(nL1Bits+1);
  numTotL1BitsAfterMcCuts.resize(nL1Bits+1);

  //resize the vectors ccording to the number of HLT paths
  numTotHltBitsBeforeCuts.resize(nHltPaths+1);
  numTotHltBitsAfterRecoCuts.resize(nHltPaths+1);
  numTotHltBitsAfterMcCuts.resize(nHltPaths+1);
  

  if (dbe) {
    dbe->setCurrentFolder(dirname_);
    dbe->rmdir(dirname_);
  }
  
  
  if (dbe) {
    dbe->setCurrentFolder(dirname_);
    }
  
  dbe_->setCurrentFolder(dirname_+triggerBitsDir);
  //add 1 bin for the Total
  hL1BitsBeforeCuts  = dbe_->book1D("L1Bits", "L1 Trigger Bits",nL1Bits+1, 0, nL1Bits+1);    
  hHltBitsBeforeCuts = dbe_->book1D("HltBits","HL Trigger Bits",nHltPaths+1, 0, nHltPaths+1);

//   hL1OverlapNormToTotal        = dbe_->book2D("L1OverlapNormToTotal"       ,"Overlap among L1 paths, norm to the Total number of Events", 1, 0, 1, 1, 0, 1);
//   hHltOverlapNormToTotal       = dbe_->book2D("HltOverlapNormToTotal"      ,"Overlap among Hlt paths, norm to the Total number of Events ", 1, 0, 1, 1, 0, 1);
//   hL1OverlapNormToLargestPath  = dbe_->book2D("L1OverlapNormToLargestPath" ,"Overlap among L1 paths, norm to the Largest of the couple ", 1, 0, 1, 1, 0, 1);
//   hHltOverlapNormToLargestPath = dbe_->book2D("HltOverlapNormToLargestPath","Overlap among Hlt paths, norm to the Largest of the couple ", 1, 0, 1, 1, 0, 1);

  dbe_->setCurrentFolder(dirname_+recoSelBitsDir);   
  hL1BitsAfterRecoCuts  = dbe_->book1D("L1Bits", "L1 Trigger Bits",nL1Bits+1, 0, nL1Bits+1);    
  hHltBitsAfterRecoCuts = dbe_->book1D("HltBits","HL Trigger Bits",nHltPaths+1, 0, nHltPaths+1);

  dbe_->setCurrentFolder(dirname_+mcSelBitsDir);   
  hL1BitsAfterMcCuts  = dbe_->book1D("L1Bits", "L1 Trigger Bits",nL1Bits+1, 0, nL1Bits+1);    
  hHltBitsAfterMcCuts = dbe_->book1D("HltBits","HL Trigger Bits",nHltPaths+1, 0, nHltPaths+1);

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

  dbe_->setCurrentFolder(dirname_+recoSelBitsDir);
  hTemp = (TH1F*) (hL1BitsAfterRecoCuts->getTH1F())->Clone("L1Paths");
  hL1PathsAfterRecoCuts   = dbe_->book1D("L1Paths", hTemp);
  hTemp = (TH1F*) (hHltBitsAfterRecoCuts->getTH1F())->Clone("HltPaths");
  hHltPathsAfterRecoCuts  = dbe_->book1D("HltPaths", hTemp);

  dbe_->setCurrentFolder(dirname_+mcSelBitsDir);
  hTemp = (TH1F*) (hL1BitsAfterMcCuts->getTH1F())->Clone("L1Paths");
  hL1PathsAfterMcCuts   = dbe_->book1D("L1Paths", hTemp);
  hTemp = (TH1F*) (hHltBitsAfterMcCuts->getTH1F())->Clone("HltPaths");
  hHltPathsAfterMcCuts  = dbe_->book1D("HltPaths", hTemp);
  
  for(unsigned int i=0; i<l1Names_.size(); ++i) {
    hL1PathsBeforeCuts->setBinLabel(i+1,l1Names_[i].c_str(),1);
    hL1PathsAfterRecoCuts->setBinLabel(i+1,l1Names_[i].c_str(),1);
    hL1PathsAfterMcCuts->setBinLabel(i+1,l1Names_[i].c_str(),1);
  }
  for (unsigned int i=0; i<hlNames_.size(); ++i) {
    hHltPathsBeforeCuts->setBinLabel(i+1,hlNames_[i].c_str(),1);
    hHltPathsAfterRecoCuts->setBinLabel(i+1,hlNames_[i].c_str(),1);
    hHltPathsAfterMcCuts->setBinLabel(i+1,hlNames_[i].c_str(),1);
  }






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



  delete myRecoSelector;
  if(mcFlag) delete myMcSelector;
  delete myPlotMaker;
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


// BeginRun
void TriggerValidator::beginRun(const edm::Run& run, const edm::EventSetup& c)
{
  LogDebug("TriggerValidator") << "beginRun, run " << run.id();
}





//define this as a plug-in
DEFINE_FWK_MODULE(TriggerValidator);
