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
// $Id: TriggerValidator.cc,v 1.2 2007/09/28 11:10:19 chiorbo Exp $
//
//


// system include files
#include <memory>
#include <stdio.h>
#include <iomanip>

#include "HLTriggerOffline/SUSYBSM/interface/TriggerValidator.h"

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


#include "TFile.h"
#include "TH1.h"
#include "TH2.h"
#include "TList.h"
#include "TObjString.h"
#include "TString.h"
#include "TObject.h"
#include "TDirectory.h"

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
  HistoFileName(iConfig.getUntrackedParameter("histoFileName",
					      std::string("SusyBsmTriggerValidation.root"))),
  StatFileName(iConfig.getUntrackedParameter("statFileName",
					      std::string("SusyBsmTriggerValidation.stat"))),
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
  iEvent.getByType(L1GTRR);
  std::vector<int> l1bits;
  if (!L1GTRR.isValid()) {cout << "L1ParticleMapCollection Not Valid!" << endl;}
  int nL1size = L1GTRR->decisionWord().size();

  // To resize the vectors and the Histo according to the number of paths, I first
  // set histo with 1 bin, and then resize only if Nbin = 1,
  // so that it's done only at the first event.
  // It could be improved if I could read the number of paths from the Setup
  // and not from the events, but I don't know how to do that.
  if(hL1BitsBeforeCuts->GetNbinsX() == 1) {

    //resize the vectors ccording to the number of L1 paths
    numTotL1BitsBeforeCuts.resize(L1GTRR->decisionWord().size()+1);
    numTotL1BitsAfterRecoCuts.resize(L1GTRR->decisionWord().size()+1);
    numTotL1BitsAfterMcCuts.resize(L1GTRR->decisionWord().size()+1);

    //rebin the eff histograms according to the number of L1 paths
    hL1BitsBeforeCuts->SetBins(L1GTRR->decisionWord().size()+1, 0, L1GTRR->decisionWord().size()+1);
    hL1BitsAfterRecoCuts->SetBins(L1GTRR->decisionWord().size()+1, 0, L1GTRR->decisionWord().size()+1);
    hL1BitsAfterMcCuts->SetBins(L1GTRR->decisionWord().size()+1, 0, L1GTRR->decisionWord().size()+1);

    //rebin the overlap histograms according to the number of L1 paths
    hL1OverlapNormToTotal->SetBins(L1GTRR->decisionWord().size(),0,L1GTRR->decisionWord().size(),L1GTRR->decisionWord().size(),0,L1GTRR->decisionWord().size());
    hL1OverlapNormToLargestPath->SetBins(L1GTRR->decisionWord().size(),0,L1GTRR->decisionWord().size(),L1GTRR->decisionWord().size(),0,L1GTRR->decisionWord().size());

    //resize the eff and overlap vectors ccording to the number of L1 paths
    effL1BeforeCuts.resize(L1GTRR->decisionWord().size()+1);
    effL1AfterRecoCuts.resize(L1GTRR->decisionWord().size()+1);
    effL1AfterMcCuts.resize(L1GTRR->decisionWord().size()+1);
    vCorrL1.resize(L1GTRR->decisionWord().size());
    for(unsigned int i=0; i<L1GTRR->decisionWord().size(); i++) {vCorrL1[i].resize(L1GTRR->decisionWord().size());}
    vCorrNormL1.resize(L1GTRR->decisionWord().size());
    for(unsigned int i=0; i<L1GTRR->decisionWord().size(); i++) {vCorrNormL1[i].resize(L1GTRR->decisionWord().size());}


    

    //Get the names of the L1 paths
    //for the moment the names are not included in L1GlobalTriggerReadoutRecord
    //we need to use L1GlobalTriggerObjectMapRecord
    edm::Handle<L1GlobalTriggerObjectMapRecord> gtObjectMapRecord;
    //    iEvent.getByLabel("l1GtEmulDigis", gtObjectMapRecord);
    iEvent.getByLabel("gtDigis", gtObjectMapRecord);
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

  //The overlap histos are filled in the endJob() method

  
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

  // To resize the vectors and the Histo according to the number of paths, I first
  // set histo with 1 bin, and then resize only if Nbin = 1,
  // so that it's done only at the first event.
  // It could be improved if I could read the number of paths from the Setup
  // and not from the events, but I don't know how to do that.

  if(hHltBitsBeforeCuts->GetNbinsX() == 1) {

    //resize the vectors ccording to the number of HLT paths
    numTotHltBitsBeforeCuts.resize(trhv->size()+1);
    numTotHltBitsAfterRecoCuts.resize(trhv->size()+1);
    numTotHltBitsAfterMcCuts.resize(trhv->size()+1);

    //rebin the eff histograms according to the number of HLT paths
    hHltBitsBeforeCuts->SetBins(trhv->size()+1, 0, trhv->size()+1);
    hHltBitsAfterRecoCuts->SetBins(trhv->size()+1, 0, trhv->size()+1);
    hHltBitsAfterMcCuts->SetBins(trhv->size()+1, 0, trhv->size()+1);


    //rebin the overlap histograms according to the number of HLT paths
    hL1HltMap->SetBins(L1GTRR->decisionWord().size(), 0, L1GTRR->decisionWord().size(), trhv->size(), 0, trhv->size());
    hHltOverlapNormToTotal->SetBins(trhv->size(),0,trhv->size(),trhv->size(),0,trhv->size());
    hHltOverlapNormToLargestPath->SetBins(trhv->size(),0,trhv->size(),trhv->size(),0,trhv->size());


    //resize the eff and overlap vectors ccording to the number of L1 paths
    effHltBeforeCuts.resize(trhv->size()+1);
    effHltAfterRecoCuts.resize(trhv->size()+1);
    effHltAfterMcCuts.resize(trhv->size()+1);
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

  //fill the histo of correletions between l1 and hlt
  for(int iL1=0; iL1<nL1size; iL1++) {
    for(unsigned int iHLT=0; iHLT<trhv->size(); iHLT++) {
      if(l1bits[iL1] && hltbits[iHLT]) hL1HltMap->Fill(iL1,iHLT);
    }
  }







  if(!alreadyBooked) {
    myPlotMaker->bookHistos(&l1bits,&hltbits,&l1Names_,&hlNames_);
    myTurnOnMaker->bookHistos();
    alreadyBooked = true;
  }
  myPlotMaker->fillPlots(iEvent);
  myTurnOnMaker->fillPlots(iEvent);

}


// ------------ method called once each job just before starting event loop  ------------
void 
TriggerValidator::beginJob(const edm::EventSetup&)
{


  myRecoSelector = new RecoSelector(userCut_params);
  if(mcFlag) myMcSelector   = new McSelector(userCut_params);
  myPlotMaker   = new PlotMaker(objectList);
  myTurnOnMaker = new TurnOnMaker(turnOn_params);
  alreadyBooked = false;

  // Initialize ROOT output file
  theHistoFile = new TFile(HistoFileName.c_str(), "RECREATE");
  theHistoFile->mkdir("TriggerBits");
  theHistoFile->mkdir("RecoSelection");
  theHistoFile->mkdir("McSelection");
  TDirectory* dirTurnOnCurves = theHistoFile->mkdir("TurnOnCurves");
  dirTurnOnCurves->mkdir("Muon");
  TDirectory* dirL1Jets = theHistoFile->mkdir("L1Jets");
  TDirectory* dirL1JetsCentral = dirL1Jets->mkdir("Central");
  TDirectory* dirL1JetsForward = dirL1Jets->mkdir("Forward");
  TDirectory* dirL1JetsTau = dirL1Jets->mkdir("Tau");
  dirL1JetsCentral->mkdir("General");
  dirL1JetsCentral->mkdir("L1"     );
  dirL1JetsCentral->mkdir("HLT"    );
  dirL1JetsForward->mkdir("General");
  dirL1JetsForward->mkdir("L1"     );
  dirL1JetsForward->mkdir("HLT"    );
  dirL1JetsTau->mkdir("General");
  dirL1JetsTau->mkdir("L1"     );
  dirL1JetsTau->mkdir("HLT"    );
  TDirectory* dirL1Em = theHistoFile->mkdir("L1Em");
  TDirectory* dirIso = dirL1Em->mkdir("Isolated");
  TDirectory* dirNotIso = dirL1Em->mkdir("NotIsolated");
  dirIso->mkdir("General");
  dirIso->mkdir("L1"     );
  dirIso->mkdir("HLT"    );
  dirNotIso->mkdir("General");
  dirNotIso->mkdir("L1"     );
  dirNotIso->mkdir("HLT"    );
  TDirectory* dirL1Muons = theHistoFile->mkdir("L1Muons");
  dirL1Muons->mkdir("General");
  dirL1Muons->mkdir("L1"     );
  dirL1Muons->mkdir("HLT"    );
  TDirectory* dirL1MET = theHistoFile->mkdir("L1MET");
  dirL1MET->mkdir("General");
  dirL1MET->mkdir("L1"     );
  dirL1MET->mkdir("HLT"    );
  TDirectory* dirRecoJets = theHistoFile->mkdir("RecoJets");
  dirRecoJets->mkdir("General");
  dirRecoJets->mkdir("L1"     );
  dirRecoJets->mkdir("HLT"    );
  TDirectory* dirRecoElectrons = theHistoFile->mkdir("RecoElectrons");
  dirRecoElectrons->mkdir("General");
  dirRecoElectrons->mkdir("L1");
  dirRecoElectrons->mkdir("HLT");
  TDirectory* dirRecoMuons = theHistoFile->mkdir("RecoMuons");
  dirRecoMuons->mkdir("General");
  dirRecoMuons->mkdir("L1");
  dirRecoMuons->mkdir("HLT");
  TDirectory* dirRecoPhotons = theHistoFile->mkdir("RecoPhotons");
  dirRecoPhotons->mkdir("General");
  dirRecoPhotons->mkdir("L1");
  dirRecoPhotons->mkdir("HLT");
  TDirectory* dirRecoMET = theHistoFile->mkdir("RecoMET");
  dirRecoMET->mkdir("General");
  dirRecoMET->mkdir("L1");
  dirRecoMET->mkdir("HLT");
  
  theHistoFile->cd("/TriggerBits");   
  //book all the histograms with only 1 bin
  //the number of bins will be changed in the analyze() method according to the number of L1 and HLT paths
  hL1BitsBeforeCuts  = new TH1D("L1Bits", "L1 Trigger Bits",1, 0, 1);
  hHltBitsBeforeCuts = new TH1D("HltBits","HL Trigger Bits",1, 0, 1);
  hL1HltMap         = new TH2D("L1HltMap", "Map of L1 and HLT bits", 1, 0, 1, 1, 0, 1);

  hL1OverlapNormToTotal        = new TH2D("L1OverlapNormToTotal"       ,"Overlap among L1 paths, norm to the Total number of Events", 1, 0, 1, 1, 0, 1);
  hHltOverlapNormToTotal       = new TH2D("HltOverlapNormToTotal"      ,"Overlap among Hlt paths, norm to the Total number of Events ", 1, 0, 1, 1, 0, 1);
  hL1OverlapNormToLargestPath  = new TH2D("L1OverlapNormToLargestPath" ,"Overlap among L1 paths, norm to the Largest of the couple ", 1, 0, 1, 1, 0, 1);
  hHltOverlapNormToLargestPath = new TH2D("HltOverlapNormToLargestPath","Overlap among Hlt paths, norm to the Largest of the couple ", 1, 0, 1, 1, 0, 1);

  theHistoFile->cd("/RecoSelection");   
  hL1BitsAfterRecoCuts  = new TH1D("L1Bits", "L1 Trigger Bits",1, 0, 1);
  hHltBitsAfterRecoCuts = new TH1D("HltBits","HL Trigger Bits",1, 0, 1);
  theHistoFile->cd("/McSelection");   
  hL1BitsAfterMcCuts  = new TH1D("L1Bits", "L1 Trigger Bits",1, 0, 1);
  hHltBitsAfterMcCuts = new TH1D("HltBits","HL Trigger Bits",1, 0, 1);
  theHistoFile->cd();   

}



// ------------ write the histograms into the root file  ------------
void 
TriggerValidator::writeHistos() {

  gDirectory->cd("/TriggerBits");
  hL1BitsBeforeCuts->Write();
  hHltBitsBeforeCuts->Write();
  hL1PathsBeforeCuts->Write();
  hHltPathsBeforeCuts->Write();
  hL1HltMapNorm->Write();
  hL1HltMapNorm->Write();
  hL1OverlapNormToTotal->Write();
  hHltOverlapNormToTotal->Write();
  hL1OverlapNormToLargestPath->Write();
  hHltOverlapNormToLargestPath->Write();
  


  gDirectory->cd("/RecoSelection");
  hL1BitsAfterRecoCuts->Write();
  hHltBitsAfterRecoCuts->Write();
  hL1PathsAfterRecoCuts->Write();
  hHltPathsAfterRecoCuts->Write();

  gDirectory->cd("/McSelection");
  hL1BitsAfterMcCuts->Write();
  hHltBitsAfterMcCuts->Write();
  hL1PathsAfterMcCuts->Write();
  hHltPathsAfterMcCuts->Write();

}


// ------------ method called once each job just after ending the event loop  ------------
void 
TriggerValidator::endJob() {


  //  myTurnOnMaker->finalOperations();

  //calculate the final efficiencies and the normalizations
  for(unsigned int i=0; i<numTotL1BitsBeforeCuts.size()-1; i++) {
    effL1BeforeCuts[i] = (double)numTotL1BitsBeforeCuts[i]/(double)nEvTot;
    for(unsigned int j=0; j<numTotL1BitsBeforeCuts.size()-1; j++) {
      vCorrNormL1[i][j] = (double)vCorrL1[i][j]/(double)nEvTot;
    }
  }

  for(unsigned int i=0; i<numTotHltBitsBeforeCuts.size()-1; i++) {
    effHltBeforeCuts[i] = (double)numTotHltBitsBeforeCuts[i]/(double)nEvTot;
    for(unsigned int j=0; j<numTotHltBitsBeforeCuts.size()-1; j++) {
      vCorrNormHlt[i][j] = (double)vCorrHlt[i][j]/(double)nEvTot;
    }
  }

  //after the reco cuts

  if(nEvRecoSelected) {
    for(unsigned int i=0; i<numTotL1BitsAfterRecoCuts.size()-1; i++) {
      effL1AfterRecoCuts[i] = (double)numTotL1BitsAfterRecoCuts[i]/(double)nEvRecoSelected;
    }
    
    for(unsigned int i=0; i<numTotHltBitsAfterRecoCuts.size()-1; i++) {
      effHltAfterRecoCuts[i] = (double)numTotHltBitsAfterRecoCuts[i]/(double)nEvRecoSelected;
    }
  } else {
    
    for(unsigned int i=0; i<numTotL1BitsAfterRecoCuts.size()-1; i++) {
      effL1AfterRecoCuts[i] = -1;
    }
    
    for(unsigned int i=0; i<numTotHltBitsAfterRecoCuts.size()-1; i++) {
      effHltAfterRecoCuts[i] = -1;
    }
  }


  //after the mc cuts
  if(nEvMcSelected) {
    for(unsigned int i=0; i<numTotL1BitsAfterMcCuts.size()-1; i++) {
      effL1AfterMcCuts[i] = (double)numTotL1BitsAfterMcCuts[i]/(double)nEvMcSelected;
    }
    
    for(unsigned int i=0; i<numTotHltBitsAfterMcCuts.size()-1; i++) {
      effHltAfterMcCuts[i] = (double)numTotHltBitsAfterMcCuts[i]/(double)nEvMcSelected;
    }
  } else {
    for(unsigned int i=0; i<numTotL1BitsAfterMcCuts.size()-1; i++) {
      effL1AfterMcCuts[i] = -1;
    }
    
    for(unsigned int i=0; i<numTotHltBitsAfterMcCuts.size()-1; i++) {
      effHltAfterMcCuts[i] = -1;
    }
  }    



  //create the histos with paths
  //identical to the ones with "bits"
  //but with the names in the x axis
  //instead of the numbers
  theHistoFile->cd("/TriggerBits");
  hL1PathsBeforeCuts  = (TH1D*) hL1BitsBeforeCuts ->Clone("L1Paths");
  hHltPathsBeforeCuts = (TH1D*) hHltBitsBeforeCuts->Clone("HltPaths");	

  theHistoFile->cd("/RecoSelection");
  hL1PathsAfterRecoCuts   = (TH1D*) hL1BitsAfterRecoCuts  ->Clone("L1Paths");
  hHltPathsAfterRecoCuts  = (TH1D*) hHltBitsAfterRecoCuts ->Clone("HltPaths"); 

  theHistoFile->cd("/McSelection");
  hL1PathsAfterMcCuts   = (TH1D*) hL1BitsAfterMcCuts  ->Clone("L1Paths");
  hHltPathsAfterMcCuts  = (TH1D*) hHltBitsAfterMcCuts ->Clone("HltPaths"); 
  
  for(unsigned int i=0; i<l1Names_.size(); ++i) {
    hL1PathsBeforeCuts->GetXaxis()->SetBinLabel(i+1,l1Names_[i].c_str());
    hL1PathsAfterRecoCuts->GetXaxis()->SetBinLabel(i+1,l1Names_[i].c_str());
    hL1PathsAfterMcCuts->GetXaxis()->SetBinLabel(i+1,l1Names_[i].c_str());
  }
  for (unsigned int i=0; i<hlNames_.size(); ++i) {
    hHltPathsBeforeCuts->GetXaxis()->SetBinLabel(i+1,hlNames_[i].c_str());
    hHltPathsAfterRecoCuts->GetXaxis()->SetBinLabel(i+1,hlNames_[i].c_str());
    hHltPathsAfterMcCuts->GetXaxis()->SetBinLabel(i+1,hlNames_[i].c_str());
  }


  //create the normalized L1-HLT map
  theHistoFile->cd("/TriggerBits");
  double normContent = 0;
  hL1HltMapNorm = (TH2D*) hL1HltMap->Clone("L1HltMapNorm");
  for(int iL1=1; iL1<hL1BitsBeforeCuts->GetNbinsX()+1; iL1++) {
    for(int iHLT=1; iHLT<hHltBitsBeforeCuts->GetNbinsX()+1; iHLT++) {
      if(hL1HltMap->GetBinContent(iL1,iHLT) == 0 && hHltBitsBeforeCuts->GetBinContent(iHLT) == 0) {
	normContent = 0;
      } else {
	normContent = hL1HltMap->GetBinContent(iL1,iHLT)/hHltBitsBeforeCuts->GetBinContent(iHLT);
      }
      hL1HltMapNorm->SetBinContent(iL1,iHLT,normContent);
    }
  }
  

  //fill the overlap histos
  for(unsigned int i=0; i<numTotL1BitsBeforeCuts.size()-1; i++) {
    for(unsigned int j=0; j<numTotL1BitsBeforeCuts.size()-1; j++) {
      int iNorm = 0;
      if(effL1BeforeCuts[i] > effL1BeforeCuts[j]) {iNorm  = i;}
      else {iNorm = j;}
      double effNorm  =  effL1BeforeCuts[iNorm]>0 ?  vCorrNormL1[i][j] / effL1BeforeCuts[iNorm]  : 0;
      hL1OverlapNormToTotal->SetBinContent(i+1,j+1, vCorrNormL1[i][j]);
      hL1OverlapNormToLargestPath->SetBinContent(i+1,j+1,effNorm);
    }
  }


  for(unsigned int i=0; i<numTotHltBitsBeforeCuts.size()-1; i++) {
    for(unsigned int j=0; j<numTotHltBitsBeforeCuts.size()-1; j++) {
      int iNorm = 0;
      if(effHltBeforeCuts[i] > effHltBeforeCuts[j]) {iNorm  = i;}
      else {iNorm = j;}
      double effNorm  = (effHltBeforeCuts[iNorm]>0) ? vCorrNormHlt[i][j]/effHltBeforeCuts[iNorm] : 0;
      hHltOverlapNormToTotal->SetBinContent(i+1,j+1, vCorrNormHlt[i][j]);
      hHltOverlapNormToLargestPath->SetBinContent(i+1,j+1,effNorm);
    }
  }






  this->writeHistos();
  myPlotMaker->writeHistos();
  myTurnOnMaker->writeHistos();


  theHistoFile->Write();
  theHistoFile->Close();

  //  using namespace std;

  unsigned int n(l1Names_.size());

  n = l1Names_.size();
  cout << endl;
  cout << "L1T-Table "
       << right << setw(10) << "L1T  Bit#" << " "
       << "Name" << "\n";
  for (unsigned int i=0; i!=n; i++) {
    cout << right << setw(20) << i << " "
	 << l1Names_[i] << "\n";
  }
  
  
  n = hlNames_.size();
  cout << endl;
  cout << "HLT-Table "
       << right << setw(10) << "HLT  Bit#" << " "
       << "Name" << "\n";
  
  for (unsigned int i=0; i!=n; i++) {
    cout << right << setw(20) << i << " "
	 << hlNames_[i] << "\n";
  }
  
  cout << endl;
  cout << "HLT-Table end!" << endl;
  cout << endl;
  

  //Print in a stat file the efficiencies and the overlaps
 
 
  ofstream statFile(StatFileName.c_str(),ios::out);


  statFile << "*********************************************************************************" << endl;
  statFile << "*********************************************************************************" << endl;
  statFile << "                                   L1 Efficiencies                               " << endl;
  statFile << "*********************************************************************************" << endl;
  statFile << "*********************************************************************************" << endl;
  statFile << endl;
  statFile << "---------------------------------------------------------------------------------" << endl;
  statFile << "---------------------------------------------------------------------------------" << endl;
  statFile << "|           L1 Path             |   eff (Tot)    | eff (Reco Sel)|  eff (Mc Sel) |" << endl;
  statFile << "---------------------------------------------------------------------------------" << endl;
  statFile << "---------------------------------------------------------------------------------" << endl;
  for(unsigned int i=0; i<numTotL1BitsBeforeCuts.size()-1; i++) {
    statFile << "|  " << left << setw(29) << l1Names_[i] << "|" << setprecision(3) << showpoint << right << setw(13) << effL1BeforeCuts[i]    << "  |" <<
                                                                                                            setw(13) << effL1AfterRecoCuts[i] << "  |" <<
                                                                                                            setw(13) << effL1AfterMcCuts[i]   << "  |" << endl;
  }
  statFile << "---------------------------------------------------------------------------------" << endl;
  statFile << "---------------------------------------------------------------------------------" << endl;
  statFile << endl;
  statFile << endl;
  statFile << endl;
  statFile << endl;
  statFile << endl;
  statFile << endl;



  statFile << "**********************************************************************************" << endl;
  statFile << "**********************************************************************************" << endl;
  statFile << "                                  Hlt Efficiencies                                " << endl;
  statFile << "**********************************************************************************" << endl;
  statFile << "**********************************************************************************" << endl;
  statFile << endl;
  statFile << "----------------------------------------------------------------------------------" << endl;
  statFile << "----------------------------------------------------------------------------------" << endl;
  statFile << "|           Hlt Path             |   eff (Tot)    | eff (Reco Sel)|  eff (Mc Sel) |" << endl;
  statFile << "----------------------------------------------------------------------------------" << endl;
  statFile << "----------------------------------------------------------------------------------" << endl;
  for(unsigned int i=0; i<numTotHltBitsBeforeCuts.size()-1; i++) {
    statFile << "|  " << left << setw(29) << hlNames_[i] << "|" << setprecision(3) << showpoint << right << setw(13) << effHltBeforeCuts[i]    << "  |" << 
                                                                                                            setw(13) << effHltAfterRecoCuts[i] << "  |" << 
                                                                                                            setw(13) << effHltAfterMcCuts[i]   << "  |" <<endl;
  }
  statFile << "----------------------------------------------------------------------------------" << endl;
  statFile << "----------------------------------------------------------------------------------" << endl;
  statFile << endl;
  statFile << endl;
  statFile << endl;
  statFile << endl;
  statFile << endl;
  statFile << endl;





  statFile << "****************************************************************************************************************************************************" << endl; 
  statFile << "****************************************************************************************************************************************************" << endl; 
  statFile << "                                                      L1 Correlations   (only overlaps >5% are shown, and only without any selection)                                               " << endl;
  statFile << "****************************************************************************************************************************************************" << endl; 
  statFile << "****************************************************************************************************************************************************" << endl;
  statFile << endl;
  statFile << "----------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
  statFile << "----------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
  statFile << "|           L1 Path 1           |           L1 Path 2           |  Overlap Norm to Total  |  Overlap Norm to Path  |         Path of Norm          |" << endl;
  statFile << "----------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
  statFile << "----------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
  statFile << endl;
  for(unsigned int i=0; i<numTotL1BitsBeforeCuts.size()-1; i++) {
    for(unsigned int j=0; j<numTotL1BitsBeforeCuts.size()-1; j++) {
      if(vCorrNormL1[i][j]>0.05) {
	int iNorm = 0;
	if(effL1BeforeCuts[i] > effL1BeforeCuts[j]) {iNorm  = i;}
	else {iNorm = j;}
	double effNorm  =  vCorrNormL1[i][j] / effL1BeforeCuts[iNorm];
	statFile << "|  " << left << setw(29) << l1Names_[i] << "|  " << setw(29) <<  left << l1Names_[j] << "|"
		 << setprecision(3) << showpoint << right  << setw(22) << vCorrNormL1[i][j] << "   |"
		 << setprecision(3) << showpoint << right  << setw(21) << effNorm           << "   |  "
		 << left << setw(29) << l1Names_[iNorm] << "|" << endl;
      }
    }
  statFile << "----------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
  }
  statFile << "----------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
  statFile << "----------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
  statFile << endl;
  statFile << endl;
  statFile << endl;
  statFile << endl;
  statFile << endl;
  statFile << endl;


  statFile << "****************************************************************************************************************************************************" << endl; 
  statFile << "****************************************************************************************************************************************************" << endl; 
  statFile << "                                                     Hlt Correlations   (only overlaps >5% are shown, and only without any selection)                                               " << endl;
  statFile << "****************************************************************************************************************************************************" << endl; 
  statFile << "****************************************************************************************************************************************************" << endl;
  statFile << endl;
  statFile << "----------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
  statFile << "----------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
  statFile << "|           Hlt Path 1          |           Hlt Path 2          |  Overlap Norm to Total  |  Overlap Norm to Path  |         Path of Norm          |" << endl;
  statFile << "----------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
  statFile << "----------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
  statFile << endl;
  for(unsigned int i=0; i<numTotHltBitsBeforeCuts.size()-1; i++) {
    for(unsigned int j=0; j<numTotHltBitsBeforeCuts.size()-1; j++) {
      if(vCorrNormHlt[i][j]>0.05) {
	int iNorm = 0;
	if(effHltBeforeCuts[i] > effHltBeforeCuts[j]) {iNorm  = i;}
	else {iNorm = j;}
	double effNorm  = vCorrNormHlt[i][j]/effHltBeforeCuts[iNorm];
	statFile << "|  " << left << setw(29) << hlNames_[i] << "|  " << setw(29) <<  left << hlNames_[j] << "|"
		 << setprecision(3) << showpoint << right  << setw(22) << vCorrNormHlt[i][j] << "   |"
		 << setprecision(3) << showpoint << right  << setw(21) << effNorm            << "   |  "
		 << left << setw(29) << hlNames_[iNorm] << "|" << endl;
      }
    }
  statFile << "----------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
  }
  statFile << "----------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
  statFile << "----------------------------------------------------------------------------------------------------------------------------------------------------" << endl;
  statFile << endl;
  statFile << endl;
  statFile << endl;
  statFile << endl;
  statFile << endl;
  statFile << endl;




  statFile.close();



  delete myRecoSelector;
  if(mcFlag) delete myMcSelector;
  delete myPlotMaker;
  delete myTurnOnMaker;
  return;
}


//define this as a plug-in
DEFINE_FWK_MODULE(TriggerValidator);
