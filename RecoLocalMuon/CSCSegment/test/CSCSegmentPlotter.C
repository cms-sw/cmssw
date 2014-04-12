void CSCSegmentPlotter(int segtype){

  /* Macro to plot histograms produced by CSCRecHitReader.cc
   * You may need to update the TFile name, and will need to
   * input the segtype as shown below.
   *
   * Author:  Dominique Fortin - UCR
   */

  float nsigmas = 1.5;    // Number of sigmas around mean to fit gaussian.  It uses 2 iterations 
                          // i.e. range is = [mu - nsigmas * sigma, mu + nsigmas * sigma]

// Files for histogram output --> set suffixps to desired file type:  e.g. .eps, .jpg, ...

TString suffixps = ".gif";

TString segment = "shower";

 if (segtype == 2) TString segment = "shower_zoom";
 if (segtype == 3) TString segment = "DF";
 if (segtype == 4) TString segment = "DF_zoom";

 TString endfile = ".root";
 TString tfile = "cscsegments_"+segment+endfile;

 TFile *file = TFile::Open(tfile);

 TString plot0 = "matched_pair_"+segment+suffixps;

 TString plot1a = "eff_raw_"+segment+suffixps;
 TString plot1b = "eff_6hit_"+segment+suffixps;

 TString plot2  = "chi2_"+segment+suffixps; 

 TString plot3  = "nhits_per_seg_"+segment+suffixps;

 TString plot4a = "dx_seg_Ori_"+segment+suffixps;
 TString plot4b = "dy_seg_Ori_"+segment+suffixps;
 TString plot4c = "dphi_seg_Dir_"+segment+suffixps;
 TString plot4d = "dtheta_seg_Dir_"+segment+suffixps;

// ********************************************************************
// Pointers to histograms
// ********************************************************************


// 1) Eff
 hRawEff     = (TH1F *) file->Get("h1");
 h6hitEff    = (TH1F *) file->Get("h3");

// 2) Chi^2
 hChi2       = (TH1F *) file->Get("h4");

// 3) Number hit/segment
 hNhit       = (TH1F *) file->Get("h5");

// 4) Resolution on segment origin (dx, dy), and direction (dphi, dtheta)
 hdxME1A     = (TH1F *) file->Get("h20"); 
 hdyME1A     = (TH1F *) file->Get("h30"); 
 hdphiME1A   = (TH1F *) file->Get("h40"); 
 hdthetaME1A = (TH1F *) file->Get("h50"); 

// *****************************************************************
// 1) Efficiency
// *****************************************************************

 // 1a) Raw efficiency
 gStyle->SetOptStat(kFALSE);
 TCanvas *c1 = new TCanvas("c1","");
 c1->SetFillColor(10);   
 c1->SetFillColor(10);
 hRawEff->SetTitle(segment);
 hRawEff->Draw();
 hRawEff->GetXaxis()->SetTitle("chamber type");
 hRawEff->GetYaxis()->SetTitle("3+ hit segment/6-layer events ");
 c1->Print(plot1a);

 // 1b) 6-hit efficiency
 gStyle->SetOptStat(kFALSE);
 TCanvas *c1 = new TCanvas("c1","");
 c1->SetFillColor(10);   
 c1->SetFillColor(10);
 h6hitEff->SetTitle(segment);
 h6hitEff->Draw();
 h6hitEff->GetXaxis()->SetTitle("chamber type");
 h6hitEff->GetYaxis()->SetTitle("6-hit segment/6-layer events ");
 c1->Print(plot1b);
 

// *****************************************************************
// 2) chi^2 
// *****************************************************************

 gStyle->SetOptStat(kTRUE);
 TCanvas *c1 = new TCanvas("c1","");
 gPad->SetLogy(kTRUE);
 c1->SetFillColor(10);   
 c1->SetFillColor(10);
 hChi2->SetTitle(segment);
 hChi2->Draw();
 hChi2->GetXaxis()->SetTitle("#chi^{2}/(2 N_{hit} - 4)");
 hChi2->GetYaxis()->SetTitle(" ");
 c1->Print(plot2);


// *****************************************************************
// 3) Number of hits per segment 
// *****************************************************************

 gStyle->SetOptStat(kTRUE);
 TCanvas *c1 = new TCanvas("c1","");
 c1->SetFillColor(10);   
 c1->SetFillColor(10);
 hNhit->SetTitle(segment);
 hNhit->Draw();
 hNhit->GetXaxis()->SetTitle("number of hits/segment");
 hNhit->GetYaxis()->SetTitle(" ");
 c1->Print(plot3);

// *****************************************************************
// 4) Resolution
// *****************************************************************

 gStyle->SetOptStat(kTRUE);
 TCanvas *c1 = new TCanvas("c1","");
 c1->SetFillColor(10);   
 c1->SetFillColor(10);
 hdxME1A->SetTitle(segment);
 hdxME1A->Draw();
 hdxME1A->GetXaxis()->SetTitle("#Delta X for seg. origin (cm)");
 hdxME1A->GetYaxis()->SetTitle(" ");
 c1->Print(plot4a);

 gStyle->SetOptStat(kTRUE);
 TCanvas *c1 = new TCanvas("c1","");
 c1->SetFillColor(10);   
 c1->SetFillColor(10);
 hdyME1A->SetTitle(segment);
 hdyME1A->Draw();
 hdyME1A->GetXaxis()->SetTitle("#Delta Y for seg. origin (cm)");
 hdyME1A->GetYaxis()->SetTitle(" ");
 c1->Print(plot4b);

 gStyle->SetOptStat(kTRUE);
 TCanvas *c1 = new TCanvas("c1","");
 c1->SetFillColor(10);   
 c1->SetFillColor(10);
 hdphiME1A->SetTitle(segment);
 hdphiME1A->Draw();
 hdphiME1A->GetXaxis()->SetTitle("#Delta #phi on seg. direction (global)");
 hdphiME1A->GetYaxis()->SetTitle(" ");
 c1->Print(plot4c);

 gStyle->SetOptStat(kTRUE);
 TCanvas *c1 = new TCanvas("c1","");
 c1->SetFillColor(10);   
 c1->SetFillColor(10);
 hdthetaME1A->SetTitle(segment);
 hdthetaME1A->Draw();
 hdthetaME1A->GetXaxis()->SetTitle("#Delta #theta on seg. direction (global)");
 hdthetaME1A->GetYaxis()->SetTitle(" ");
 c1->Print(plot4d);


 gROOT->ProcessLine(".q");

}
