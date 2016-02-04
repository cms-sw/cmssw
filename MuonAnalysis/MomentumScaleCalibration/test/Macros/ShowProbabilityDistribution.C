{
  TFile * file = new TFile("MuonAnalysis/MomentumScaleCalibration/test/Probs_new_Horace_CTEQ_1000.root", "READ");
  TH2D * hist;
  file->GetObject("GLZ0", hist);
  hist->SetTitle("Z shape from Horace x gaussian");

  hist->GetXaxis()->SetTitle("mass (GeV)");
  hist->GetXaxis()->SetTitleOffset(1.5);
  hist->GetXaxis()->SetRangeUser(80, 100);

  hist->GetYaxis()->SetTitle("resolution #sigma (GeV)");
  hist->GetYaxis()->SetTitleOffset(1.5);
  hist->GetYaxis()->SetRangeUser(0, 5);

  gStyle->SetPalette(1);
  // hist->SetContour(28);
  hist->SetContour(50);
  hist->Draw("surf5");
}
