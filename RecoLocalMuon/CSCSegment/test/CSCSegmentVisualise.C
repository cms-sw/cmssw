void CSCSegmentVisualise(int nHisto){

  /* Macro to plot histograms produced by CSCRecHitVisualise.cc
   * You may need to update the TFile name, and will need to
   * input the segtype as shown below.
   *
   * Author:  Dominique Fortin - UCR
   */


// Files for histogram output --> set suffixps to desired file type:  e.g. .eps, .jpg, ...

TString suffixps = ".gif";

 TString endfile = ".root";
 TString tfile = "cscsegments_plot"+endfile;

 TFile *file = TFile::Open(tfile);

for (int i = nHisto; i < nHisto+1; ++i ) {

  // ********************************************************************
  // Pointers to histograms
  // ********************************************************************
  char a[4];
  int j = i + 100;
 
  sprintf(a, "h%d", i+100);
  TString idx = a;

  TString plot1 = "x_vs_z_"+idx+suffixps;
  TString plot2 = "y_vs_z_"+idx+suffixps;

  // 1) rechit x vs z
  sprintf(a, "h%d", i+100);
  hxvsz = (TH2F *) file->Get(a);
 
  // 2) rechit y vs z
  sprintf(a, "h%d", i+200);
  hyvsz = (TH2F *) file->Get(a);

  // 3) used hit on seg x vs z
  sprintf(a, "h%d", i+300);
  hxvszS = (TH2F *) file->Get(a);

  // 4) used hit on seg y vs z
  sprintf(a, "h%d", i+400);
  hyvszS = (TH2F *) file->Get(a);

  // 5) Projected segment x vs z
  sprintf(a, "h%d", i+500);
  hxvszP = (TH2F *) file->Get(a);

  // 6) Projected segment y vs z
  sprintf(a, "h%d", i+600);
  hyvszP = (TH2F *) file->Get(a);

  // 7) rechit from electron x vs z
  sprintf(a, "h%d", i+700);
  hxvszE = (TH2F *) file->Get(a);
 
  // 8) rechit from electron y vs z
  sprintf(a, "h%d", i+800);
  hyvszE = (TH2F *) file->Get(a);


  // Make plot
  gStyle->SetOptStat(kFALSE);
  TCanvas *c1 = new TCanvas("c1","");
  gStyle->SetOptStat(kFALSE);
  c1->SetFillColor(10);
  c1->SetFillColor(10);
  // segments
  hxvszP->SetMarkerSize(0.2);
  hxvszP->SetMarkerStyle(6);
  hxvszP->SetMarkerColor(kRed);
//  hxvszP->SetTitle("X-local vs Z-local");
  hxvszP->GetXaxis()->SetTitle("local z (cm)");
  hxvszP->GetYaxis()->SetTitle("local x (cm)");
  hxvszP->Draw();
  // rechits
  hxvsz->SetMarkerSize(3);
  hxvsz->SetMarkerStyle(30);
  hxvsz->SetMarkerColor(kBlack);
  hxvsz->Draw("SAME");
  // rechits from electrons
  hxvszE->SetMarkerSize(3);
  hxvszE->SetMarkerStyle(29);
  hxvszE->SetMarkerColor(kGreen);
  hxvszE->Draw("SAME");
  // used hits on segments
  hxvszS->SetMarkerSize(2);
  hxvszS->SetMarkerStyle(29);
  hxvszS->SetMarkerColor(kBlue);
  hxvszS->Draw("SAME");
  c1->Print(plot1);  


  // Make plot
  gStyle->SetOptStat(kFALSE);
  TCanvas *c1 = new TCanvas("c1","");
  gStyle->SetOptStat(kFALSE);
  c1->SetFillColor(10);
  c1->SetFillColor(10);
  // segments
  hyvszP->SetMarkerSize(0.2);
  hyvszP->SetMarkerStyle(6);
  hyvszP->SetMarkerColor(kRed);
//  hyvszP->SetTitle("Y-local vs Z-local");
  hyvszP->GetXaxis()->SetTitle("local z (cm)");
  hyvszP->GetYaxis()->SetTitle("local y (cm)");
  hyvszP->Draw();
  // rechits
  hyvsz->SetMarkerSize(3);
  hyvsz->SetMarkerStyle(30);
  hyvsz->SetMarkerColor(kBlack);
  hyvsz->Draw("SAME");
  // rechits from electrons
  hyvszE->SetMarkerSize(3);
  hyvszE->SetMarkerStyle(29);
  hyvszE->SetMarkerColor(kGreen);
  hyvszE->Draw("SAME");
  // used hits on segments
  hyvszS->SetMarkerSize(2);
  hyvszS->SetMarkerStyle(29);
  hyvszS->SetMarkerColor(kBlue);
  hyvszS->Draw("SAME");
  c1->Print(plot2);  

}
 
gROOT->ProcessLine(".q");

}
