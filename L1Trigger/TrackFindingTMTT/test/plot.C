{
  // In unnamed scripts, variables not forgotten at end, so must delete them before rerunning script, so ...
  gROOT->Reset("a");
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat("");
  //gStyle->SetOptStat("emr");
  //  gStyle->SetOptStat("euom");
  gStyle->SetStatFontSize(0.035);
  gStyle->SetHistFillColor(kBlue);
  gStyle->SetHistFillStyle(1001);
  gStyle->SetMarkerSize(2.0);

  gStyle->SetStatFormat("5.3f");
  gStyle->SetStatFontSize(0.04);
  gStyle->SetOptFit(0111);
  gStyle->SetStatW(0.30);
  gStyle->SetStatH(0.02);
  gStyle->SetStatX(0.9);
  gStyle->SetStatY(0.9);
  gStyle->SetPadLeftMargin(0.15);
  gStyle->SetTitleYOffset(1.3);
  gStyle->SetTitleSize(0.05, "XYZ");

  gStyle->SetLabelSize(.04,"x");
  gStyle->SetLabelSize(.04,"y");

  gStyle->SetCanvasDefH(500);
  gStyle->SetCanvasDefW(800);

  TCanvas d1("d1");

  TFile *file[7];
  
  /*
  file[1] = new TFile("out_ttbar_ultimate_off1_20180831_183411/Hist.root"); // Corr 1 off
  file[2] = new TFile("out_ttbar_ultimate_off2_20180831_183030/Hist.root"); // Corr 2 off
  file[3] = new TFile("out_ttbar_ultimate_off3_20180831_183117/Hist.root"); // Corr 3 off
  file[4] = new TFile("out_ttbar_ultimate_off4_20180831_183220/Hist.root"); // Corr 4 off
  file[5] = new TFile("out_ttbar_ultimate_20180831_182908/Hist.root");  // All on
  file[6] = new TFile("out_ttbar_ultimate_offall_20180831_183257/Hist.root"); // All off
  TLegend leg(0.7,0.15,0.9,0.45);    
  */

  // Testing variants
  /*
  file[1] = new TFile("out_ttbar_ultimate_option1_20180831_183601/Hist.root");  // All on
  file[2] = new TFile("out_ttbar_ultimate_approxB_20180831_183534/Hist.root");  // All on
  file[3] = new TFile("out_ttbar_ultimate_dsbydr1_20180901_230848/Hist.root"); // Corr 3 off
  //  file[4] = new TFile("out_ttbar_ultimate_option1_deltaS0_way1_20180901_231217/Hist.root"); // Corr 4 off - worse resolution near eta = 1.5
  file[4] = new TFile("out_ttbar_ultimate_option1_deltaS0_way2_20180901_231405//Hist.root"); // Corr 4 off - Even worse resolution near eta = 1.5 - 1.7
  file[5] = new TFile("out_ttbar_ultimate_20180831_182908/Hist.root");  // All on
  file[6] = new TFile("out_ttbar_ultimate_offall_20180831_183257/Hist.root"); // All off
  TLegend leg(0.7,0.15,0.9,0.45);    
  */

  /*
  // Different particle types
  file[1] = new TFile("out_ttbar_ultimate_onlyE_20180903_115714/Hist.root"); 
  file[2] = new TFile("out_ttbar_ultimate_noE_20180903_115836/Hist.root");  
  file[3] = new TFile("out_ttbar_ultimate_20180831_182908/Hist.root");  
  d1.SetLogy(1);  
  TLegend leg(0.7,0.15,0.9,0.45);    
  */

  file[1] = new TFile("out_muon_ultimate_off1_20180903_145832/Hist.root"); // Corr 1 off
  file[2] = new TFile("out_muon_ultimate_off2_20180903_145511/Hist.root"); // Corr 2 off
  file[3] = new TFile("out_muon_ultimate_off3_20180903_145549/Hist.root"); // Corr 3 off
  file[4] = new TFile("out_muon_ultimate_off4_20180903_145621/Hist.root"); // Corr 4 off
  file[5] = new TFile("out_muon_ultimate_20180903_145432/Hist.root");  // All on
  file[6] = new TFile("out_muon_ultimate_offall_20180903_145655/Hist.root"); // All off
  TLegend leg(0.2,0.6,0.4,0.9);

  TString name[7] = {"", "Corr. 1 off", "Corr. 2 off", "Corr. 3 off", "Corr. 4 off", "All on", "All off"};
  //TString name[7] = {"", "e", "#mu, #pi, K, p", "e, #mu, #pi, K, p", "Corr. 4 off", "All on", "All off"};
  unsigned int icol[7] = {0, 1, 2, 3, 6, 8, 9};

  TH1F* his;
  TH2F* his2D;
  TProfile *prof[7];
  TEfficiency *teffi1, *teffi2, *teffi3;

  float ymax = 0.;

  bool first = true;
  for (unsigned int i = 1; i <= 6; i++) {

    //if (i > 3) continue;

    file[i]->GetObject("TMTrackProducer/KF4ParamsComb/QoverPtResVsTrueEta_KF4ParamsComb", prof[i]);
    //file[i]->GetObject("TMTrackProducer/KF4ParamsComb/QoverPtResVsTrueInvPt_KF4ParamsComb", prof[i]);
    //file[i]->GetObject("TMTrackProducer/KF4ParamsComb/Z0ResVsTrueEta_KF4ParamsComb", prof[i]);
    //file[i]->GetObject("TMTrackProducer/KF4ParamsComb/FitChi2DofVsInvPtMatched_KF4ParamsComb", prof[i]);
    //file[i]->GetObject("TMTrackProducer/KF4ParamsComb/FitChi2DofVsInvPtUnmatched_KF4ParamsComb", prof[i]);
    //file[i]->GetObject("TMTrackProducer/KF4ParamsComb/FitChi2DofVsEtaMatched_KF4ParamsComb", prof[i]);
    //file[i]->GetObject("TMTrackProducer/KF4ParamsComb/FitChi2DofVsEtaUnmatched_KF4ParamsComb", prof[i]);

    float ym = prof[i]->GetMaximum();
    if (ymax < ym) ymax = ym;
    prof[i]->SetMaximum(1.8*ymax);
    //prof[i]->SetMaximum(0.02);
    prof[i]->SetMinimum(0.0);

    if (prof[i] == nullptr) {
      cout<<"ERROR: Input histogram missing "<<i<<endl;
      cin.get();
      continue;
    }

    prof[i]->SetMarkerStyle(20 + i);
    prof[i]->SetMarkerColor(icol[i]);
    if (first) {
      first = false;
      prof[i]->Draw("P ");
    } else {
      prof[i]->Draw("P SAME");
    }
    leg.AddEntry(prof[i], name[i], "P");
    leg.Draw();
    d1.Draw(); d1.Update(); 
  }
 
  //prof1->SetTitle(";1/Pt (1/GeV); #phi_{0} resolution");
  
  d1.Print("plot.pdf");
  cin.get(); 

  for (unsigned int i = 1; i <= 6; i++) {
    file[i]->Close();
    delete file[i];
  }
}
