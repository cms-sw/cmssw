// stack macro
// Pasquale Noli

#include <sstream>
#include <string>
#include "stack_common.h"

void stack_zGolden() {
  const int rebin = 1;
  int nbins = 0;
  // sum oh 1HLT+2HLT+1HLTAB
  TH1F *hZ_1HLT = (TH1F*)z.Get("goodZToMuMu1HLTPlots/zMass");
  TH1F *hZ_2HLT = (TH1F*)z.Get("goodZToMuMu2HLTPlots/zMass");
  TH1F *hZ_1HLTAB = (TH1F*)z.Get("goodZToMuMuOnlyOneMuonWithEtaLessThan2p11HLTPlots/zMass");
  nbins = hZ_1HLT->GetNbinsX();
  TH1F *h1 = new TH1F ("h1", "h1", nbins, 0, nbins);
  for(size_t i =0; i < nbins; ++i){
    int val = hZ_1HLT->GetBinContent(i) + hZ_2HLT->GetBinContent(i) + hZ_1HLTAB->GetBinContent(i);
    h1->SetBinContent(i, val);
  }
  
  TH1F *hW_1HLT = (TH1F*)w.Get("goodZToMuMu1HLTPlots/zMass");
  TH1F *hW_2HLT = (TH1F*)w.Get("goodZToMuMu2HLTPlots/zMass");
  TH1F *hW_1HLTAB = (TH1F*)w.Get("goodZToMuMuOnlyOneMuonWithEtaLessThan2p11HLTPlots/zMass");
  nbins = hW_1HLT->GetNbinsX();
  TH1F *h2 = new TH1F ("h2", "h2", nbins, 0, nbins);
  for (size_t i =0; i < nbins; ++i){
    val = hW_1HLT->GetBinContent(i) + hW_2HLT->GetBinContent(i) + hW_1HLTAB->GetBinContent(i);
    h2->SetBinContent(i , val );
  }
  
  TH1F *hT_1HLT = (TH1F*)tt.Get("goodZToMuMu1HLTPlots/zMass");
  TH1F *hT_2HLT = (TH1F*)tt.Get("goodZToMuMu2HLTPlots/zMass");
  TH1F *hT_1HLTAB = (TH1F*)tt.Get("goodZToMuMuOnlyOneMuonWithEtaLessThan2p11HLTPlots/zMass");
  nbins = hT_1HLT->GetNbinsX();
  TH1F *h3 = new TH1F ("h3", "h3", nbins, 0, nbins);
  for ( size_t i =0 ; i<nbins ; i++  ){
    val = hT_1HLT->GetBinContent(i) + hT_2HLT->GetBinContent(i) + hT_1HLTAB->GetBinContent(i);
    h3->SetBinContent(i , val );
  }
  
  TH1F *hQ_1HLT = (TH1F*)qcd.Get("goodZToMuMu1HLTPlots/zMass");
  TH1F *hQ_2HLT = (TH1F*)qcd.Get("goodZToMuMu2HLTPlots/zMass");
  TH1F *hQ_1HLTAB = (TH1F*)qcd.Get("goodZToMuMuOnlyOneMuonWithEtaLessThan2p11HLTPlots/zMass");
  nbins = hQ_1HLT->GetNbinsX();
  TH1F *h4 = new TH1F ("h4", "h4", nbins, 0, nbins);
  for ( size_t i =0 ; i<nbins ; i++  ){
    int val = hQ_1HLT->GetBinContent(i) + hQ_2HLT->GetBinContent(i) + hQ_1HLTAB->GetBinContent(i);
    h4->SetBinContent(i , val );
  }
  TH1F *hdata = 0;


  makeStack(h1, h2, h3, h4, hdata, 0.0001, rebin);

  stat(h1, h2, h3, h4, hdata);

  c1->SaveAs("zGolden");
}
