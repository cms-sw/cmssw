void CSCGainsStudy_MakePlots(int debug){
  
  /** Macro to plot histograms produced by CSCGainsStudy.cc
   *
   * Author:  Dominique Fortin - UCR
   */
  
  
  TFile *file = TFile::Open("csc_strip_gains.root");
  
  // set suffixps to desired file type:  e.g. .eps, .jpg, ...
  TString suffixps = ".jpg";
  
  
  for (int i = 1; i < 37; i++ ) {
    int j = 0;
    if (  j == i ) TString chamber = "All_CSC";
    if (++j == i ) TString chamber = "ME_11_27";
    if (++j == i ) TString chamber = "ME_11_28";
    if (++j == i ) TString chamber = "ME_11_29";
    if (++j == i ) TString chamber = "ME_11_30";
    if (++j == i ) TString chamber = "ME_11_31";
    if (++j == i ) TString chamber = "ME_11_32";
    if (++j == i ) TString chamber = "ME_12_27";
    if (++j == i ) TString chamber = "ME_12_28";
    if (++j == i ) TString chamber = "ME_12_29";
    if (++j == i ) TString chamber = "ME_12_30";
    if (++j == i ) TString chamber = "ME_12_31";
    if (++j == i ) TString chamber = "ME_12_32";
    if (++j == i ) TString chamber = "ME_13_27";
    if (++j == i ) TString chamber = "ME_13_28";
    if (++j == i ) TString chamber = "ME_13_29";
    if (++j == i ) TString chamber = "ME_13_30";
    if (++j == i ) TString chamber = "ME_13_31";
    if (++j == i ) TString chamber = "ME_13_32";
    if (++j == i ) TString chamber = "ME_21_14";
    if (++j == i ) TString chamber = "ME_21_15";
    if (++j == i ) TString chamber = "ME_21_16";
    if (++j == i ) TString chamber = "ME_22_27";
    if (++j == i ) TString chamber = "ME_22_28";
    if (++j == i ) TString chamber = "ME_22_29";
    if (++j == i ) TString chamber = "ME_22_30";
    if (++j == i ) TString chamber = "ME_22_31";
    if (++j == i ) TString chamber = "ME_22_32";
    if (++j == i ) TString chamber = "ME_31_14";
    if (++j == i ) TString chamber = "ME_31_15";
    if (++j == i ) TString chamber = "ME_31_16";
    if (++j == i ) TString chamber = "ME_32_27";
    if (++j == i ) TString chamber = "ME_32_28";
    if (++j == i ) TString chamber = "ME_32_29";
    if (++j == i ) TString chamber = "ME_32_30";
    if (++j == i ) TString chamber = "ME_32_31";
    if (++j == i ) TString chamber = "ME_32_32";


    // Set pointers to histograms
    hGains = (TH1F *) file->Get("hGain_"+chamber);
    hGaindiff = (TH1F *) file->Get("hGaindiff_"+chamber); 
    hGainvsch = (TH2F *) file->Get("hGainvsch_"+chamber);

    gStyle->SetOptFit(0111);

    // 1) weight
    TString plot1 = "strip_weight_"+chamber+suffixps;
    gStyle->SetOptStat(kTRUE);
    TCanvas *c1 = new TCanvas("c1","");
    c1->SetFillColor(10);
    c1->SetLogy(1);
    hGains->Draw();
    c1->Print(plot1); 
    
 
    // 2) weight difference
    TString plot2 = "delta_strip_weight_"+chamber+suffixps;
    gStyle->SetOptStat(kTRUE);
    TCanvas *c1 = new TCanvas("c1","");
    c1->SetLogy(1);
    c1->SetFillColor(10);
    hGaindiff->Draw();
    hGaindiff->Fit("gaus");
    c1->Print(plot2);

    if (i > 0) {
       // 3) strip weight vs channel #
       TString plot3 = "strip_weight_vs_channel_"+chamber+suffixps;
       gStyle->SetOptStat(kTRUE);
       TCanvas *c1 = new TCanvas("c1","");
       c1->SetLogy(0);
       c1->SetFillColor(10);
       hGainvsch->Draw("BOX");
       c1->Print(plot3); 
    }
  }

//  gROOT->ProcessLine(".q");
  
}
