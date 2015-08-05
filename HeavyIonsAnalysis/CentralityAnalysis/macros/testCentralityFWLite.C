#include <string>
#include <vector>
#include <iostream>

#include <TH2D.h>
#include <TH1D.h>
#include <TFile.h>
#include <TSystem.h>



#if !defined(__CINT__) && !defined(__MAKECINT__)

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/FWLite/interface/Event.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/HeavyIonEvent/interface/CentralityBins.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/HeavyIonEvent/interface/Centrality.h"
#include "SimDataFormats/HiGenData/interface/GenHIEvent.h"

#endif

using namespace std;


void testCentralityFWLite(){

   TFile * centFile = new TFile("../data/CentralityTables.root");
   TFile* infile = new TFile("/net/hisrv0001/home/yetkin/pstore02/ana/Hydjet_MinBias_d20100222/DEF33D38-12E8-DE11-BA8F-0019B9CACF1A.root");
  fwlite::Event event(infile);
  TFile* outFile = new TFile("test.root","recreate");

  TH1D::SetDefaultSumw2();
  TH2D* hNpart = new TH2D("hNpart",";Npart Truth;Npart RECO",50,0,500,50,0,500);
  TH1D* hBins = new TH1D("hBins",";bins;events",44,-1,21);

  CentralityBins::RunMap HFhitBinMap = getCentralityFromFile(centFile,"makeCentralityTableTFile", "HFhitsAMPT_2760GeV", 149500, 155000);

  // loop the events
  unsigned int iEvent=0;
  for(event.toBegin(); !event.atEnd(); ++event, ++iEvent){
     edm::EventBase const & ev = event;
    if( iEvent % 10 == 0 ) cout<<"Processing event : "<<iEvent<<endl;
    edm::Handle<edm::GenHIEvent> mc;
    ev.getByLabel(edm::InputTag("heavyIon"),mc);
    edm::Handle<reco::Centrality> cent;
    ev.getByLabel(edm::InputTag("hiCentrality"),cent);

    double b = mc->b();
    double npart = mc->Npart();
    double ncoll = mc->Ncoll();
    double nhard = mc->Nhard();

    double hf = cent->EtHFhitSum();
    double hftp = cent->EtHFtowerSumPlus();
    double hftm = cent->EtHFtowerSumMinus();
    double eb = cent->EtEBSum();
    double eep = cent->EtEESumPlus();
    double eem = cent->EtEESumMinus();

    int run = ev.id().run();

    int bin = HFhitBinMap[run]->getBin(hf);
    hBins->Fill(bin);

    double npartMean = HFhitBinMap[run]->NpartMean(hf);
    double npartSigma = HFhitBinMap[run]->NpartSigma(hf);
    hNpart->Fill(npart,npartMean);
  }

  outFile->cd();
  hBins->Write();
  hNpart->Write();
  outFile->Write();
  
}
