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
//         Created:  Wed Aug 29 15:10:56 CEST 2007
// $Id$
//
//


// system include files
#include <memory>


#include "HLTriggerOffline/SUSYBSM/interface/TriggerValidator.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Handle.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"

//Added by Max for the Trigger
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/L1Trigger/interface/L1ParticleMap.h"


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
  userCut_params(iConfig.getParameter<ParameterSet>("UserCutParams")),
  objectList(iConfig.getParameter<ParameterSet>("ObjectList"))
{
   //now do what ever initialization is needed
  theHistoFile = 0;
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


   bool eventSelected = myCutSelector->isSelected(iEvent);
   //   cout << "risultato selector = " << (int)myCutSelector->isSelected(iEvent) << endl;
   

  // ******************************************************** 
  // Get the L1 Info
  // ********************************************************    
  Handle<l1extra::L1ParticleMapCollection> L1PMC;
  try {iEvent.getByType(L1PMC);} catch (...) {;}
  std::vector<int> l1bits;
  if (!L1PMC.isValid()) {cout << "L1ParticleMapCollection Not Valid!" << endl;}
  int nL1size = L1PMC->size();

  if(hL1BitsBeforeCuts->GetNbinsX() == 1) {
    hL1BitsBeforeCuts->SetBins(L1PMC->size()+1, 0, L1PMC->size()+1);
    numTotL1BitsBeforeCuts.reserve(L1PMC->size()+1);
    hL1BitsAfterCuts->SetBins(L1PMC->size()+1, 0, L1PMC->size()+1);
    numTotL1BitsAfterCuts.reserve(L1PMC->size()+1);
    l1Names_.resize(L1PMC->size()+1);
    for (unsigned int i=0; i!=L1PMC->size(); i++) {    
	l1extra::L1ParticleMap::L1TriggerType 
	  type(static_cast<l1extra::L1ParticleMap::L1TriggerType>(i));
	l1Names_[i]=l1extra::L1ParticleMap::triggerName(type);
    }
    l1Names_[L1PMC->size()] = "Total";
  }

  for (int i=0; i<nL1size; ++i) {
    l1bits.push_back((*L1PMC)[i].triggerDecision());
    if((*L1PMC)[i].triggerDecision()) {
      numTotL1BitsBeforeCuts[i]++;
      hL1BitsBeforeCuts->Fill(i);
      if(eventSelected) {
	numTotL1BitsAfterCuts[i]++;
	hL1BitsAfterCuts->Fill(i);
      }
    }      
  }

  numTotL1BitsBeforeCuts[nL1size]++;
  hL1BitsBeforeCuts->Fill(nL1size);
  if(eventSelected) {
    numTotL1BitsAfterCuts[nL1size]++;
    hL1BitsAfterCuts->Fill(nL1size);
  }

  // ******************************************************** 
  // Get the HLT Info
  // ******************************************************** 
  vector<Handle<TriggerResults> > trhv;
  iEvent.getManyByType(trhv);
  const unsigned int n(trhv.size());
  //  if(n>1) cout << "More than one TriggerResult Object! Please check you are using the correct one" << endl;
  std::vector<int> hltbits;
  int iRefHlt = 0;
  for(unsigned int i=0; i<n; i++) {
    if((*(trhv[i])).size()>1) {
      if(iRefHlt) cout << "WARNING: more than one TriggerResult Object with multiple bits!!!!!!!" << endl;
      iRefHlt = i;
    }
  }
//   cout << "(*(trhv[0])).size() = " << (*(trhv[0])).size() << endl;
//   cout << "(*(trhv[1])).size() = " << (*(trhv[1])).size() << endl;
//   cout << "(*(trhv[2])).size() = " << (*(trhv[2])).size() << endl;

  if(hHltBitsBeforeCuts->GetNbinsX() == 1) {
    hHltBitsBeforeCuts->SetBins((*(trhv[iRefHlt])).size()+1, 0, (*(trhv[iRefHlt])).size()+1);
    numTotHltBitsBeforeCuts.reserve((*(trhv[iRefHlt])).size()+1);
    hHltBitsAfterCuts->SetBins((*(trhv[iRefHlt])).size()+1, 0, (*(trhv[iRefHlt])).size()+1);
    hL1HltMap->SetBins(L1PMC->size(), 0, L1PMC->size(), (*(trhv[iRefHlt])).size(), 0, (*(trhv[iRefHlt])).size());
   numTotHltBitsAfterCuts.reserve((*(trhv[iRefHlt])).size()+1);
    hlNames_=(*(trhv[iRefHlt])).getTriggerNames();
    hlNames_.push_back("Total");
  }

  for(unsigned int i=0; i< (*(trhv[iRefHlt])).size(); i++) {
    hltbits.push_back((*(trhv[iRefHlt])).at(i).accept());
    if((*(trhv[iRefHlt])).at(i).accept()) {
      numTotHltBitsBeforeCuts[i]++;
      hHltBitsBeforeCuts->Fill(i);
      if(eventSelected) {
	numTotHltBitsAfterCuts[i]++;
	hHltBitsAfterCuts->Fill(i);
      }
    }      
  }

  numTotHltBitsBeforeCuts[(*(trhv[iRefHlt])).size()]++;
  hHltBitsBeforeCuts->Fill((*(trhv[iRefHlt])).size());
  if(eventSelected) {
    numTotHltBitsAfterCuts[(*(trhv[iRefHlt])).size()]++;
    hHltBitsAfterCuts->Fill((*(trhv[iRefHlt])).size());
  }


  for(unsigned int iL1=0; iL1<nL1size; iL1++) {
    for(unsigned int iHLT=0; iHLT<(*(trhv[iRefHlt])).size(); iHLT++) {
      if(l1bits[iL1] && hltbits[iHLT]) hL1HltMap->Fill(iL1,iHLT);
    }
  }


  if(!alreadyBooked) {
    myPlotMaker->bookHistos(&l1bits,&hltbits,&l1Names_,&hlNames_);
    alreadyBooked = true;
  }
  myPlotMaker->fillPlots(iEvent);


}


// ------------ method called once each job just before starting event loop  ------------
void 
TriggerValidator::beginJob(const edm::EventSetup&)
{

  myCutSelector = new CutSelector(userCut_params);
  myPlotMaker   = new PlotMaker(objectList);
  alreadyBooked = false;

  // Initialize ROOT output file
   theHistoFile = new TFile(HistoFileName.c_str(), "RECREATE");
   theHistoFile->mkdir("TriggerBits");
   theHistoFile->mkdir("Selection");
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
   hL1BitsBeforeCuts  = new TH1D("L1Bits", "L1 Trigger Bits",1, 0, 1);
   hHltBitsBeforeCuts = new TH1D("HltBits","HL Trigger Bits",1, 0, 1);
   hL1HltMap         = new TH2D("L1HltMap", "Map of L1 and HLT bits", 1, 0, 1, 1, 0, 1);
   theHistoFile->cd("/Selection");   
   hL1BitsAfterCuts  = new TH1D("L1Bits", "L1 Trigger Bits",1, 0, 1);
   hHltBitsAfterCuts = new TH1D("HltBits","HL Trigger Bits",1, 0, 1);
   theHistoFile->cd();   

//    lL1Names = new TList();
//    lHLTNames = new TList();

}

// ------------ method called once each job just after ending the event loop  ------------
void 
TriggerValidator::endJob() {

  theHistoFile->cd("/TriggerBits");
  hL1PathsBeforeCuts  = (TH1D*) hL1BitsBeforeCuts ->Clone("L1Paths");
  hHltPathsBeforeCuts = (TH1D*) hHltBitsBeforeCuts->Clone("HltPaths");	

  theHistoFile->cd("/Selection");
  hL1PathsAfterCuts   = (TH1D*) hL1BitsAfterCuts  ->Clone("L1Paths");
  hHltPathsAfterCuts  = (TH1D*) hHltBitsAfterCuts ->Clone("HltPaths"); 

  
  for(int i=0; i<l1Names_.size(); ++i) {
    hL1PathsBeforeCuts->GetXaxis()->SetBinLabel(i+1,l1Names_[i].c_str());
    hL1PathsAfterCuts->GetXaxis()->SetBinLabel(i+1,l1Names_[i].c_str());
  }
  for (int i=0; i<hlNames_.size(); ++i) {
    hHltPathsBeforeCuts->GetXaxis()->SetBinLabel(i+1,hlNames_[i].c_str());
    hHltPathsAfterCuts->GetXaxis()->SetBinLabel(i+1,hlNames_[i].c_str());
  }


  theHistoFile->cd("/TriggerBits");
  double normContent = 0;
  hL1HltMapNorm = (TH2D*) hL1HltMap->Clone("L1HltMapNorm");
  for(unsigned int iL1=1; iL1<hL1BitsBeforeCuts->GetNbinsX()+1; iL1++) {
    for(unsigned int iHLT=1; iHLT<hHltBitsBeforeCuts->GetNbinsX()+1; iHLT++) {
      if(hL1HltMap->GetBinContent(iL1,iHLT) == 0 && hHltBitsBeforeCuts->GetBinContent(iHLT) == 0) {
	normContent = 0;
      } else {
	normContent = hL1HltMap->GetBinContent(iL1,iHLT)/hHltBitsBeforeCuts->GetBinContent(iHLT);
      }
      hL1HltMapNorm->SetBinContent(iL1,iHLT,normContent);
    }
  }
  
  theHistoFile->cd();
  
  
  theHistoFile->Write();
  theHistoFile->Close() ;

//   delete lL1Names;
//   delete lHLTNames;

  using namespace std;

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
  


  delete myCutSelector;
  delete myPlotMaker;
  return;
}

//define this as a plug-in
DEFINE_FWK_MODULE(TriggerValidator);
