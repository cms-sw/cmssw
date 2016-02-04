void printGctValidationHistos()
{
  TFile *f = new TFile("gctValidationPlots.root");

  TPDF *outf = new TPDF("gctValidationHistos.pdf");

  TCanvas *c1 = new TCanvas;

  gStyle->SetPalette(1);

  c1->Divide(2,2);
  c1->cd(1); Plot(f,"l1GctValidation/L1GctEtSums/SumEtInGeV","Total Et (GeV)"); c1->Update();
  c1->cd(2); Plot(f,"l1GctValidation/L1GctEtSums/SumHtInGeV","Total Ht (GeV)"); c1->Update();
  c1->cd(3); Plot(f,"l1GctValidation/L1GctEtSums/SumEtInLsb","Total Et (L1 units)"); c1->Update();
  c1->cd(4); Plot(f,"l1GctValidation/L1GctEtSums/SumHtInLsb","Total Ht (L1 units)"); c1->Update();

  c1->Clear();

  c1->Divide(2,2);
  c1->cd(1); Plot(f,"l1GctValidation/L1GctEtSums/SumEtVsInputRegions","Total Et (GeV)", "Total Et from input regions"); c1->Update();
  c1->cd(2); Plot(f,"l1GctValidation/L1GctEtSums/MissEtMagVsInputRegions","Missing Et (GeV)", "Missing Et from input regions"); c1->Update();
  c1->cd(3); Plot(f,"l1GctValidation/L1GctEtSums/MissEtAngleVsInputRegions","Mising Et phi", "Missing Et phi from input regions"); c1->Update();
  c1->cd(4); Plot(f,"l1GctValidation/L1GctEtSums/MissHtMagVsInputRegions","Missing Ht", "Missing Et from input regions"); c1->Update();

  c1->Clear();

  ((TH1*) f->Get("l1GctValidation/L1GctEtSums/MissEtAngle"))->SetMinimum(0);

  c1->Divide(3,4);
  c1->cd(1);  Plot(f,"l1GctValidation/L1GctEtSums/MissEtInGeV","Missing Et (GeV)"); c1->Update();
  c1->cd(2);  Plot(f,"l1GctValidation/L1GctEtSums/MissEtAngle","Missing Et phi (radians)"); c1->Update();
  c1->cd(4);  Plot(f,"l1GctValidation/L1GctEtSums/MissEtInLsb","Missing Et (L1 units)"); c1->Update();
  c1->cd(5);  Plot(f,"l1GctValidation/L1GctEtSums/MissEtVector","Missing Ex (GeV)", "Missing Ey (GeV)", "col"); c1->Update();
  c1->cd(7);  Plot(f,"l1GctValidation/L1GctEtSums/MissHtInGeV","Missing Et (GeV)"); c1->Update();
  c1->cd(8);  Plot(f,"l1GctValidation/L1GctEtSums/MissHtAngle","Missing Et phi (radians)"); c1->Update();
  c1->cd(10); Plot(f,"l1GctValidation/L1GctEtSums/MissHtInLsb","Missing Et (L1 units)"); c1->Update();
  c1->cd(11); Plot(f,"l1GctValidation/L1GctEtSums/MissHtVector","Missing Ex (GeV)", "Missing Ey (GeV)", "col"); c1->Update();

  c1->cd(3);  Plot(f,"l1GctValidation/L1GctEtSums/MissEtVsMissHt","Missing Et (GeV)", "Missing Ht (GeV)", "col"); c1->Update();
  c1->cd(6);  Plot(f,"l1GctValidation/L1GctEtSums/MissEtVsMissHtAngle","Missing Et phi", "Missing Ht phi", "col"); c1->Update();
  c1->cd(9);  Plot(f,"l1GctValidation/L1GctEtSums/theDPhiVsMissEt","(Met phi)-(Mht phi)", "Missing Et mag", "col"); c1->Update();
  c1->cd(12); Plot(f,"l1GctValidation/L1GctEtSums/theDPhiVsMissHt","(Met phi)-(Mht phi)", "Missing Ht mag", "col"); c1->Update();

  c1->Clear();

  c1->Divide(3,2);
  c1->cd(1); Plot(f,"l1GctValidation/L1GctEtSums/HtVsInternalJetsSum","Total Ht", "Scalar sum of jet Et"); c1->Update();
  c1->cd(2); Plot(f,"l1GctValidation/L1GctEtSums/MissHtVsInternalJetsSum","Missing Ht", "Vector sum of jet Et"); c1->Update();
  c1->cd(3); Plot(f,"l1GctValidation/L1GctEtSums/MissHtPhiVsInternalJetsSum","Missing Ht", "Vector sum of jet Et"); c1->Update();
  c1->cd(4); Plot(f,"l1GctValidation/L1GctEtSums/MissHxVsInternalJetsSum","x component of missing Ht", "Sum of jet Ex"); c1->Update();
  c1->cd(5); Plot(f,"l1GctValidation/L1GctEtSums/MissHyVsInternalJetsSum","y component of missing Ht", "Sum of jet Ey"); c1->Update();

  c1->Clear();

  gStyle->SetOptLogy(1);
  c1->Divide(4,2);
  c1->cd(1); Plot(f,"l1GctValidation/L1GctHfSumsAndJetCounts/HfRing0EtSumPositiveEta","Hf Inner Ring Et"); c1->Update();
  c1->cd(2); Plot(f,"l1GctValidation/L1GctHfSumsAndJetCounts/HfRing0EtSumNegativeEta","Hf Inner Ring Et"); c1->Update();
  c1->cd(3); Plot(f,"l1GctValidation/L1GctHfSumsAndJetCounts/HfRing1EtSumPositiveEta","Hf Inner Ring Et"); c1->Update();
  c1->cd(4); Plot(f,"l1GctValidation/L1GctHfSumsAndJetCounts/HfRing1EtSumNegativeEta","Hf Inner Ring Et"); c1->Update();
  c1->cd(5); Plot(f,"l1GctValidation/L1GctHfSumsAndJetCounts/HfRing0CountPositiveEta","Hf feature bits"); c1->Update();
  c1->cd(6); Plot(f,"l1GctValidation/L1GctHfSumsAndJetCounts/HfRing0CountNegativeEta","Hf feature bits"); c1->Update();
  c1->cd(7); Plot(f,"l1GctValidation/L1GctHfSumsAndJetCounts/HfRing1CountPositiveEta","Hf feature bits"); c1->Update();
  c1->cd(8); Plot(f,"l1GctValidation/L1GctHfSumsAndJetCounts/HfRing1CountNegativeEta","Hf feature bits"); c1->Update();
  gStyle->SetOptLogy(0);

//   c1->Clear();

//   c1->Divide(2,3);

//   //gPad->SetLogy(1);
//   c1->cd(1); Plot(f,"l1GctValidation/L1GctHfSumsAndJetCounts/JetCount#0","Count"); c1->Update();
//   c1->cd(2); Plot(f,"l1GctValidation/L1GctHfSumsAndJetCounts/JetCount#1","Count"); c1->Update();
//   c1->cd(3); Plot(f,"l1GctValidation/L1GctHfSumsAndJetCounts/JetCount#2","Count"); c1->Update();
//   c1->cd(4); Plot(f,"l1GctValidation/L1GctHfSumsAndJetCounts/JetCount#3","Count"); c1->Update();
//   c1->cd(5); Plot(f,"l1GctValidation/L1GctHfSumsAndJetCounts/JetCount#4","Count"); c1->Update();
//   c1->cd(6); Plot(f,"l1GctValidation/L1GctHfSumsAndJetCounts/JetCount#5","Count"); c1->Update();

  outf->Close();

}

void Plot(TFile* f, TString Hist, TString XAxisLabel, TString YAxisLabel="Events", TString Opt="")
{

  // Get the histograms from the files
  TH1D *H   = (TH1D*)f->Get(Hist);

  // Add the X axis label
  H->GetXaxis()->SetTitle(XAxisLabel);
  H->GetYaxis()->SetTitle(YAxisLabel);

  H->SetFillColor(kRed);
  //H->SetMinimum(0);
  //H->GetXaxis()->SetRangeUser(0.0, 200.0);

  // plot 
  H->Draw(Opt);

}
