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

TString segment = "DF";

 if (segtype == 1) TString segment = "SK";
 if (segtype == 3) TString segment = "DF";
 if (segtype == 4) TString segment = "ST";

 TString endfile = ".root";
 TString tfile = "cscsegments_plot"+endfile;

 TFile *file = TFile::Open(tfile);

 TString plot0 = "matched_pair_"+segment+suffixps;

 TString plot1a = "eff_raw_"+segment+suffixps;
 TString plot1b = "eff_6hit_"+segment+suffixps;

 TString plot2  = "chi2_"+segment+suffixps; 

 TString plot3  = "nhits_per_seg_"+segment+suffixps;


// ********************************************************************
// Pointers to histograms
// ********************************************************************


// 1) Eff
 hRawEff     = (TH1F *) file->Get("h1");
 h6hitEff    = (TH1F *) file->Get("h3");

// 2) Chi^2
 hChi2       = (TH1F *) file->Get("h4");

// 5) Number hit/segment
 hNhit       = (TH1F *) file->Get("h5");

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
 c1->SetFillColor(10);   
 c1->SetFillColor(10);
 hChi2->SetTitle(segment);
 hChi2->Draw();
 hChi2->GetXaxis()->SetTitle("#chi^{2}");
 hChi2->GetYaxis()->SetTitle(" ");
 c1->Print(plot2);


// *****************************************************************
// 2) Number of hits per segment 
// *****************************************************************

 gStyle->SetOptStat(kTRUE);
 TCanvas *c1 = new TCanvas("c1","");
 c1->SetFillColor(10);   
 c1->SetFillColor(10);
 hNhit->SetTitle(segment);
 hNhit->Draw();
 hNhit->GetXaxis()->SetTitle("#chi^{2}");
 hNhit->GetYaxis()->SetTitle(" ");
 c1->Print(plot3);



 gROOT->ProcessLine(".q");

}
