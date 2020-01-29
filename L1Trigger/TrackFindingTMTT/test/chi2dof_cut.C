// Plot fraction of good & bad tracks surviving as function of chi2 cut.
// Does so for subset of tracks with nSkip skipped layers (or if nSkip=-1 for all tracks).

void chi2dof_cut(int nSkip) {
  //=== Optimise chi2 cut applied to fitted tracks.

  // In unnamed scripts, variables not forgotten at end, so must delete them before rerunning script, so ...
  gROOT->Reset("a");
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat("");
  //gStyle->SetOptStat("emr");
  //  gStyle->SetOptStat("euom");
  gStyle->SetStatFontSize(0.035);
  gStyle->SetHistFillColor(kBlue);
  gStyle->SetHistFillStyle(1001);
  gStyle->SetMarkerSize(1.2);

  gStyle->SetStatFormat("5.3f");
  gStyle->SetStatFontSize(0.04);
  gStyle->SetOptFit(0111);
  gStyle->SetStatW(0.30);
  gStyle->SetStatH(0.02);
  gStyle->SetStatX(0.9);
  gStyle->SetStatY(0.9);
  gStyle->SetPadLeftMargin(0.20);
  gStyle->SetTitleYOffset(1.6);
  gStyle->SetTitleSize(0.05, "XYZ");

  gStyle->SetLabelSize(.04,"x");
  gStyle->SetLabelSize(.04,"y");

  gStyle->SetCanvasDefH(600);
  gStyle->SetCanvasDefW(600);

  TCanvas d1("d1");

  TH1F* hisMatchTot;
  TH1F* hisUnmatchTot;
  TH1F* hisMatch;
  TH1F* hisUnmatch;

  //TFile file1("out_ttbar_ultimate_20180831_182908/Hist.root");
  //TFile file1("out_ttbar_ultimate_offall_20180831_183257/Hist.root");
  TFile file1("Hist.root");

  TLegend leg(0.25,0.15,0.55,0.30);
  file1.GetObject("TMTrackProducer/KF4ParamsComb/FitChi2DofMatched_KF4ParamsComb", hisMatchTot);
  file1.GetObject("TMTrackProducer/KF4ParamsComb/FitChi2DofUnmatched_KF4ParamsComb", hisUnmatchTot);
  if (nSkip == 0) {
    file1.GetObject("TMTrackProducer/KF4ParamsComb/KalmanChi2DofSkipLay0Matched_KF4ParamsComb", hisMatch);
    file1.GetObject("TMTrackProducer/KF4ParamsComb/KalmanChi2DofSkipLay0Unmatched_KF4ParamsComb", hisUnmatch);
  } else if (nSkip == 1) {
    file1.GetObject("TMTrackProducer/KF4ParamsComb/KalmanChi2DofSkipLay1Matched_KF4ParamsComb", hisMatch);
    file1.GetObject("TMTrackProducer/KF4ParamsComb/KalmanChi2DofSkipLay1Unmatched_KF4ParamsComb", hisUnmatch);
  } else if (nSkip == 2) {
    file1.GetObject("TMTrackProducer/KF4ParamsComb/KalmanChi2DofSkipLay2Matched_KF4ParamsComb", hisMatch);
    file1.GetObject("TMTrackProducer/KF4ParamsComb/KalmanChi2DofSkipLay2Unmatched_KF4ParamsComb", hisUnmatch);
  } else if (nSkip == -1) {
    file1.GetObject("TMTrackProducer/KF4ParamsComb/FitChi2DofMatched_KF4ParamsComb", hisMatch);
    file1.GetObject("TMTrackProducer/KF4ParamsComb/FitChi2DofUnmatched_KF4ParamsComb", hisUnmatch);
  }
  //file1.GetObject("TMTrackProducer/KF4ParamsComb/FitBeamChi2DofMatched_KF4ParamsComb", hisMatch);
  //file1.GetObject("TMTrackProducer/KF4ParamsComb/FitBeamChi2DofUnmatched_KF4ParamsComb", hisUnmatch);

  TH1* hisMatchCum = hisMatch->GetCumulative(false);
  TH1* hisUnmatchCum = hisUnmatch->GetCumulative(false);
  unsigned int nBins = hisMatchCum->GetNbinsX();

  // TH1::GetCumulative() ignores overflow bins, so add them by hand.
  float overMatch = hisMatchCum->GetBinContent(nBins+1);
  float overUnmatch = hisUnmatchCum->GetBinContent(nBins+1);
  float lastMatch = hisMatchCum->GetBinContent(nBins);
  float lastUnmatch = hisUnmatchCum->GetBinContent(nBins);
  hisMatchCum->SetBinContent(nBins, lastMatch + overMatch);
  hisUnmatchCum->SetBinContent(nBins, lastUnmatch + overUnmatch);

  hisMatchCum->Scale(1./hisMatchTot->GetEntries());
  hisUnmatchCum->Scale(1./hisUnmatchTot->GetEntries());
  hisMatchCum->SetMarkerStyle(20);
  hisUnmatchCum->SetMarkerStyle(24);
  hisMatchCum->SetAxisRange(0.9, 35, "X"); // dangerous. don't make lower cut too big.
  hisMatchCum->SetTitle("; chi2/dof; efficiency loss");
  d1.SetLogx(1);
  d1.SetLogy(1);
  float ymax = max(hisMatchCum->GetMaximum(), hisUnmatchCum->GetMaximum());
  hisMatchCum->SetMaximum(2*ymax);
  hisMatchCum->SetMarkerStyle(20);
  hisMatchCum->Draw("P");
  leg.AddEntry(hisMatchCum,"match","P");
  hisUnmatchCum->SetMarkerStyle(24);
  hisUnmatchCum->Draw("P SAME");
  leg.AddEntry(hisUnmatchCum,"unmatch","P");
  leg.Draw();

  d1.Update();
  cin.get(); 

  TGraph graph_killed(nBins);
  graph_killed.SetTitle("; Good tracks killed; Fake tracks killed");
  for (unsigned int i = 1; i <= nBins + 1; i++) {
    float fracMatch   = hisMatchCum->GetBinContent(i);
    float fracUnmatch = hisUnmatchCum->GetBinContent(i);
    graph_killed.SetPoint(i-1, fracMatch, fracUnmatch); 
  } 
  graph_killed.Draw("AP");
 
  d1.Update();
  d1.Print("plot.pdf");
  cin.get(); 

  file1.Close();
}
