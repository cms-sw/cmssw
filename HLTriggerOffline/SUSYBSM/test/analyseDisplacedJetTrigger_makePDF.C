{
  //===============================================================
  // Make plots in PDF format using root files produced by
  // analyseDisplacedJetTrigger_cfg.py ,
  // Specify the root files obtained from running on data and
  // on MC below.
  //==============================================================
  gROOT->Reset("a");
  gStyle->SetOptTitle(0);
  gStyle.SetOptStat(0);
  gStyle->SetTitleSize(0.055, "XYZ");
  gStyle->SetLabelSize(.045,"x");
  gStyle->SetLabelSize(.045,"y");
  gStyle->SetTitleXOffset(0.9);
  gStyle->SetPadLeftMargin(0.18);
  gStyle->SetTitleYOffset(1.5);
  gStyle->SetStatY(0.9);
  gStyle->SetHistLineWidth(2);

  const bool pause = false;
  // Creation of PDF files.
  const bool printSeparatePdf = true;
  const bool printMergedPdf = false;

  //  TFile d("data.root");
  TFile d("datanew.root");
  TFile mc("mcnew.root");

  TCanvas c1;

  if (printMergedPdf) {
    gSystem->Exec("rm paper.pdf");
    c1.Print("paper.pdf[","pdf");
  }

  TH1F* his1;
  TH1F* his2;

  //
  // Number of prompt tracks in data
  //
  const float nPromptTk_cut = 2.5; // Trigger cut on number of prompt tracks per jet.
  d.cd();
  c1.SetLogy(1);
  gDirectory->GetObject("DQMData/Run 1/HLT_HT250_DoubleDisplacedJet60_v2/Run summary/recoJetNpromptTk",his1);
  gDirectory->GetObject("DQMData/Run 1/HLT_HT250_DoubleDisplacedJet60_v2/Run summary/recoJetNpromptTkMatched",his2);

  his1->Draw("HIST");
  his2->Draw("HISTSAME");
  Double_t maxHis = his1.GetMaximum();
  his1->SetMaximum(1.5*maxHis);
  his1->SetMinimum(0.9);
  his1->SetAxisRange(-0.5,15.5,"X");
  his2->SetFillColor(2);
  his1->SetXTitle("Number of prompt tracks");
  his1->SetYTitle("Number of jets");

  TLine line(nPromptTk_cut, his1->GetMinimum(), nPromptTk_cut, his1->GetMaximum());
  line.SetLineWidth(5);
  line.SetLineColor(28);
  line.SetLineStyle(7);
  line.Draw("SAME");

  c1.Draw(); c1.Update(); if (pause) {cout<<"Continue ?"<<endl; cin.get();} // Wait/sleep until use hits any key.
  if (printSeparatePdf) c1.Print("nPromptTk_Data.pdf");
  if (printMergedPdf) c1.Print("paper.pdf","pdf");
  
  //
  // Number of prompt tracks in MC
  //
  mc.cd();
  c1.SetLogy(1);
  gDirectory->GetObject("DQMData/Run 1/HLT_HT250_DoubleDisplacedJet60_v2/Run summary/recoJetNpromptTk",his1);
  gDirectory->GetObject("DQMData/Run 1/HLT_HT250_DoubleDisplacedJet60_v2/Run summary/recoJetNpromptTkMatched",his2);

  his1->Draw("HIST");
  his2->Draw("HISTSAME");
  Double_t maxHis = his1.GetMaximum();
  his1->SetMaximum(1.5*maxHis);
  his1->SetMinimum(0.9);
  his1->SetAxisRange(-0.5,15.5,"X");
  his2->SetFillColor(2);
  his1->SetXTitle("Number of prompt tracks");
  his1->SetYTitle("Number of jets");

  TLine line(nPromptTk_cut, his1->GetMinimum(), nPromptTk_cut, his1->GetMaximum());
  line.SetLineWidth(5);
  line.SetLineColor(28);
  line.SetLineStyle(7);
  line.Draw("SAME");

  c1.Draw(); c1.Update(); if (pause) {cout<<"Continue ?"<<endl; cin.get();} // Wait/sleep until use hits any key.
  if (printSeparatePdf) c1.Print("nPromptTk_MC.pdf");
  if (printMergedPdf) c1.Print("paper.pdf","pdf");

  //
  // Jet Pt in data
  //
  const float jetPt_cut = 60.; // Trigger cut on jet Pt.
  d.cd();
  c1.SetLogy(0);
  gDirectory->GetObject("DQMData/Run 1/HLT_HT250_DoubleDisplacedJet60_v2/Run summary/recoJetPt",his1);
  gDirectory->GetObject("DQMData/Run 1/HLT_HT250_DoubleDisplacedJet60_v2/Run summary/recoJetPtMatched",his2);

  his1->Draw("HIST");
  his2->Draw("HISTSAME");
  Double_t maxHis = his1.GetMaximum();
  his1->SetMaximum(1.1*maxHis);
  his1->SetMinimum(0.0);
  his1->SetAxisRange(0.,400.,"X");
  his2->SetFillColor(2);
  his1->SetXTitle("Jet Pt (GeV)");
  his1->SetYTitle("Number of jets");

  TLine line(jetPt_cut, his1->GetMinimum(), jetPt_cut, his1->GetMaximum());
  line.SetLineWidth(5);
  line.SetLineColor(28);
  line.SetLineStyle(7);
  line.Draw("SAME");

  c1.Draw(); c1.Update(); if (pause) {cout<<"Continue ?"<<endl; cin.get();} // Wait/sleep until use hits any key.
  if (printSeparatePdf) c1.Print("JetPt_Data.pdf");
  if (printMergedPdf) c1.Print("paper.pdf","pdf");

  //
  // Jet Pt in MC
  //
  mc.cd();
  c1.SetLogy(0);
  gDirectory->GetObject("DQMData/Run 1/HLT_HT250_DoubleDisplacedJet60_v2/Run summary/recoJetPt",his1);
  gDirectory->GetObject("DQMData/Run 1/HLT_HT250_DoubleDisplacedJet60_v2/Run summary/recoJetPtMatched",his2);

  his1->Draw("HIST");
  his2->Draw("HISTSAME");
  Double_t maxHis = his1.GetMaximum();
  his1->SetMaximum(1.1*maxHis);
  his1->SetMinimum(0.0);
  his1->SetAxisRange(0.,400.,"X");
  his2->SetFillColor(2);
  his1->SetXTitle("Jet Pt (GeV)");
  his1->SetYTitle("Number of jets");

  TLine line(jetPt_cut, his1->GetMinimum(), jetPt_cut, his1->GetMaximum());
  line.SetLineWidth(5);
  line.SetLineColor(28);
  line.SetLineStyle(7);
  line.Draw("SAME");

  c1.Draw(); c1.Update(); if (pause) {cout<<"Continue ?"<<endl; cin.get();} // Wait/sleep until use hits any key.
  if (printSeparatePdf) c1.Print("JetPt_MC.pdf");
  if (printMergedPdf) c1.Print("paper.pdf","pdf");

  //
  // PV z in data with 24cm vs 15.9 cm cut
  //
  gStyle.SetOptStat(1110);
  d.cd();
  c1.SetLogy(0);
  gDirectory->GetObject("DQMData/Run 1/HLT_HT250_DoubleDisplacedJet60_v2/Run summary/PVzPassed",his1);

  his1->Draw("HIST");
  Double_t maxHis = his1.GetMaximum();
  his1->SetMaximum(1.3*maxHis);
  his1->SetMinimum(0.9);
  //  his1->SetAxisRange(0.,400.,"X");
  his1->SetFillColor(2);
  his1->SetXTitle("Z of primary vertex (cm)");
  his1->SetYTitle("Number of triggered events");

  c1.Draw(); c1.Update(); if (pause) {cout<<"Continue ?"<<endl; cin.get();} // Wait/sleep until use hits any key.
  if (printSeparatePdf) c1.Print("PVz_24cmCut_Data.pdf");
  if (printMergedPdf) c1.Print("paper.pdf","pdf");

  d.cd();
  c1.SetLogy(0);
  gDirectory->GetObject("DQMData/Run 1/HLT_HT250_DoubleDisplacedJet60_v1/Run summary/PVzPassed",his2);

  his2->Draw("HIST");
  Double_t maxHis = his2.GetMaximum();
  his2->SetMaximum(1.3*maxHis);
  his2->SetMinimum(0.0);
  //  his2->SetAxisRange(0.,400.,"X");
  his2->SetFillColor(2);
  his2->SetXTitle("Z of primary vertex (cm)");
  his2->SetYTitle("Number of triggered events");

  c1.Draw(); c1.Update(); if (pause) {cout<<"Continue ?"<<endl; cin.get();} // Wait/sleep until use hits any key.
  if (printSeparatePdf) c1.Print("PVz_16cmCut_Data.pdf");
  if (printMergedPdf) c1.Print("paper.pdf","pdf");

  //
  // Number of primary vertices in data.
  //
  gStyle.SetOptStat(0);
  d.cd();
  c1.SetLogy(1);
  gDirectory->GetObject("DQMData/Run 1/HLT_HT250_DoubleDisplacedJet60_v2/Run summary/nPV",his1);
  gDirectory->GetObject("DQMData/Run 1/HLT_HT250_DoubleDisplacedJet60_v2/Run summary/nPVPassed",his2);

  his1->Draw("HIST");
  his2->Draw("HISTSAME");
  Double_t maxHis = his1.GetMaximum();
  his1->SetMaximum(1.1*maxHis);
  his1->SetMinimum(0.9);
  //  his1->SetAxisRange(0.,400.,"X");
  his2->SetFillColor(2);
  his1->SetXTitle("Number of primary vertices");
  his1->SetYTitle("Number of events");

  c1.Draw(); c1.Update(); if (pause) {cout<<"Continue ?"<<endl; cin.get();} // Wait/sleep until use hits any key.
  if (printSeparatePdf) c1.Print("nPV_Data.pdf");
  if (printMergedPdf) c1.Print("paper.pdf","pdf");

  c1.SetLogy(0);
  TGraphAsymmErrors g(his2,his1);
  g.GetXaxis()->SetLimits(-0.5, 7.5);
  float high = TMath::MaxElement(5, g.GetY()); 
  g.SetMaximum(1.5*high);
  g.Draw("AP");
  g.GetXaxis()->SetTitle("Number of primary vertices");
  g.GetYaxis()->SetTitle("Fraction of triggered events");

  c1.Draw(); c1.Update(); if (pause) {cout<<"Continue ?"<<endl; cin.get();} // Wait/sleep until use hits any key.
  if (printSeparatePdf) c1.Print("nPVeffi_Data.pdf");
  if (printMergedPdf) c1.Print("paper.pdf","pdf");

  //
  // Signal efficiency vs. decay length
  //
  mc.cd();
  c1.SetLogy(0);
  c1.SetLogx(1);
  gDirectory->GetObject("DQMData/Run 1/HLT_HT250_DoubleDisplacedJet60_v2/Run summary/trueJetProdRadius",his1);
  gDirectory->GetObject("DQMData/Run 1/HLT_HT250_DoubleDisplacedJet60_v2/Run summary/trueJetProdRadiusMatched",his2);

  TGraphAsymmErrors g(his2,his1);
  g.Draw("AP");
  //  Double_t maxHis = his1.GetMaximum();
  //  his1->SetMaximum(1.1*maxHis);
  //  his1->SetMinimum(0.9);
  g.GetXaxis()->SetLimits(0.1,40.);
  //  his2->SetFillColor(2);
  g.GetXaxis()->SetTitle("Transverse decay length of exotic (cm)");
  g.GetYaxis()->SetTitle("Trigger efficiency for jet");

  c1.Draw(); c1.Update(); if (pause) {cout<<"Continue ?"<<endl; cin.get();} // Wait/sleep until use hits any key.
  if (printSeparatePdf) c1.Print("decayLengthEffi_MC.pdf");
  if (printMergedPdf) c1.Print("paper.pdf","pdf");

  // END

  if (printMergedPdf) c1.Print("paper.pdf]");

  d.Close();
  mc.Close();
}
