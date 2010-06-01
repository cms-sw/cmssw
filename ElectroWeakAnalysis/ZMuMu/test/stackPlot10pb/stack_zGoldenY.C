
// stack macro
// Pasquale Noli

#include <sstream>
#include <string>
#include "stack_common.h"

void stack_zGoldenY() {
  const int rebin = 5;
  int nbins = 0;
  // sum oh 1HLT+2HLT+1HLTAB
  TH1F *hZ_1HLT = (TH1F*)z.Get("goodZToMuMu1HLTPlots/zRapidity");
  TH1F *hZ_2HLT = (TH1F*)z.Get("goodZToMuMu2HLTPlots/zRapidity");
  TH1F *hZ_1HLTAB = (TH1F*)z.Get("goodZToMuMuOnlyOneMuonWithEtaLessThan2p11HLTPlots/zRapidity");
  nbins = hZ_1HLT->GetNbinsX();
  //cout << nbins <<endl;
  int min = hZ_1HLT->GetXaxis()->GetXmin();
  int max = hZ_1HLT->GetXaxis()->GetXmax();
  TH1F *h1 = new TH1F ("h1", "h1", nbins/2, min/2, max/2);
  h1->GetXaxis()->SetRangeUser(-3,3);
  //  cout << 0.25 * nbins      <<endl;
 for(size_t i= 0.25 *nbins; i < 0.75 * nbins; ++i){
    int val = hZ_1HLT->GetBinContent(i) + hZ_2HLT->GetBinContent(i) + hZ_1HLTAB->GetBinContent(i);
    h1->SetBinContent(i - (0.25 *nbins), val);
  }


  TH1F *hW_1HLT = (TH1F*)w.Get("goodZToMuMu1HLTPlots/zRapidity");
  TH1F *hW_2HLT = (TH1F*)w.Get("goodZToMuMu2HLTPlots/zRapidity");
  TH1F *hW_1HLTAB = (TH1F*)w.Get("goodZToMuMuOnlyOneMuonWithEtaLessThan2p11HLTPlots/zRapidity");
  nbins = hW_1HLT->GetNbinsX();
  TH1F *h2 = new TH1F ("h2", "h2", nbins/2, min/2, max/2);
    h2->GetXaxis()->SetRangeUser(-3,3);
  for (size_t i =0.25 *nbins; i  <  0.75 * nbins; ++i){
    val = hW_1HLT->GetBinContent(i) + hW_2HLT->GetBinContent(i) + hW_1HLTAB->GetBinContent(i);
    h2->SetBinContent(i - (0.25 *nbins), val );
  }


  TH1F *hT_1HLT = (TH1F*)tt.Get("goodZToMuMu1HLTPlots/zRapidity");
  TH1F *hT_2HLT = (TH1F*)tt.Get("goodZToMuMu2HLTPlots/zRapidity");
  TH1F *hT_1HLTAB = (TH1F*)tt.Get("goodZToMuMuOnlyOneMuonWithEtaLessThan2p11HLTPlots/zRapidity");
  nbins = hT_1HLT->GetNbinsX();
  TH1F *h3 = new TH1F ("h3", "h3", nbins/2, min/2,max/2);
 h3->GetXaxis()->SetRangeUser(-3,3);
  for ( size_t i =0.25 * nbins ; i< 0.75 * nbins ; i++  ){
    val = hT_1HLT->GetBinContent(i) + hT_2HLT->GetBinContent(i) + hT_1HLTAB->GetBinContent(i);
    h3->SetBinContent(i- (0.25 *nbins) , val );
  }


  TH1F *hQ_1HLT = (TH1F*)qcd.Get("goodZToMuMu1HLTPlots/zRapidity");
  TH1F *hQ_2HLT = (TH1F*)qcd.Get("goodZToMuMu2HLTPlots/zRapidity");
  TH1F *hQ_1HLTAB = (TH1F*)qcd.Get("goodZToMuMuOnlyOneMuonWithEtaLessThan2p11HLTPlots/zRapidity");
  nbins = hQ_1HLT->GetNbinsX();
  TH1F *h4 = new TH1F ("h4", "h4", nbins/2, min/2, max/2);
 h4->GetXaxis()->SetRangeUser(-3,3);
  for ( size_t i =0.25 * nbins ; i< 0.75 * nbins ; i++  ){
    int val = hQ_1HLT->GetBinContent(i) + hQ_2HLT->GetBinContent(i) + hQ_1HLTAB->GetBinContent(i);
    h4->SetBinContent(i - (0.25 *nbins), val );
  }

  /*
  TH1F *hD_1HLT = (TH1F*)data.Get("goodZToMuMu1HLTPlots/zRapidity");
  TH1F *hD_2HLT = (TH1F*)data.Get("goodZToMuMu2HLTPlots/zRapidity");
  TH1F *hD_1HLTAB = (TH1F*)data.Get("goodZToMuMuAB1HLTPlots/zRapidity");
  nbins = hD_1HLT->GetNbinsX();
  TH1F *hdata = new TH1F ("hdata", "hdata", nbins, 0, nbins);
  for ( size_t i =0 ; i<nbins ; i++  ){
    int val = hD_1HLT->GetBinContent(i) + hD_2HLT->GetBinContent(i) + hD_1HLTAB->GetBinContent(i);
    hdata->SetBinContent(i , val );
  }
  */
  TChain chain("Events"); 
  chain.Add("NtupleLoose_135149.root");
  chain.Add("NtupleLoose_136033.root");
  chain.Add("NtupleLoose_136087.root");
  chain.Add("NtupleLoose_136100.root");
  //  TFile *datafile = TFile::Open("NtupleLoose_135149.root");
  //TTree * Events = dynamic_cast< TTree *> (datafile->Get("Events"));
  TH1F *hdata = new TH1F ("hdata", "hdata", nbins/2, min/2, max/2); 
 chain.Project("hdata", "zGoldenY" );
  hdata->Rebin(rebin); 



  makeStack(h1, h2, h3, h4, hdata, 0.001, rebin);
  hs->GetXaxis()->SetTitle("Y_{#mu #mu}");
  string yTag = "events/5 "; // use the correct rebin
  hs->GetYaxis()->SetTitle(yTag.c_str());
 
  c1->SaveAs("zGoldenY.eps");
  c1->SaveAs("zGoldenY_r5.gif");
}

//  LocalWords:  HLTAB
