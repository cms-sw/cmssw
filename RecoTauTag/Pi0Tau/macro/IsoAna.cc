#define IsoAna_cxx
#include "IsoAna.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TMath.h>
#include <TH1F.h>
#include <TCanvas.h>

void IsoAna::Loop(TString ds) {

  if (fChain == 0) return;

  Long64_t nentries = fChain->GetEntriesFast();
  Long64_t nbytes = 0, nb = 0;

  const int NWRT = 2;
  TString wrt[NWRT] = {"reco","iso"};

  TH1F* tauEt[NWRT];
  TH1F* tauEta[NWRT];
  for(int i=0; i<NWRT; i++){
    tauEt[i]  = new TH1F("tauEt_"+wrt[i],"Track Isolation Efficiency;tauEt (GeV);Efficiency",20,0.0,100.0);
    tauEta[i] = new TH1F("tauEta_"+wrt[i],"Track Isolation Efficiency;#eta;Efficiency",50,-2.5,2.5);

    tauEt[i]->Sumw2();
    tauEta[i]->Sumw2();
  }

  for (Long64_t jentry=0; jentry<nentries;jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;

    float et = momentum->Et();
    float eta = momentum->Eta();

    if(TMath::Abs(eta) > 2.5) continue;
    if(et < 15.0) continue;

    tauEt[0]->Fill(et);
    tauEta[0]->Fill(eta);

    if(nIsolationTracks == 0){
      tauEt[1]->Fill(et);
      tauEta[1]->Fill(eta);
    }

  }// for jentry


  TCanvas* can_et = new TCanvas("can_et_"+ds,"can_et_"+ds,800,600);
  TH1F* isoEt = (TH1F*) tauEt[0]->Clone("isoEt_"+ds);
  isoEt->Divide(tauEt[1],tauEt[0],1.0,1.0,"B");
  isoEt->SetMarkerStyle(20);
  isoEt->Draw();
  can_et->Print("isoEt_"+ds+".gif","gif");

  TCanvas* can_eta = new TCanvas("can_eta_"+ds,"can_eta_"+ds,800,600);
  TH1F* isoEta = (TH1F*) tauEta[0]->Clone("isoEta_"+ds);
  isoEta->Divide(tauEta[1],tauEta[0],1.0,1.0,"B");
  isoEta->SetMarkerStyle(20);
  isoEta->Draw();
  can_eta->Print("isoEta_"+ds+".gif","gif");

}
