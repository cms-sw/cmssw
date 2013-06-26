{
  double xMin = 2.5;
  double xMax = 3.5;

  TFile * file = new TFile("3_MuScleFit.root");

  TProfile * probHisto = file->FindObjectAny("Mass_fine_PProf");
  TH1F * histo = file->FindObjectAny("hRecBestResAllEvents_Mass");
  histo->Rebin(4);

  // Compute integrals
  Int_t histoBinMin = histo->FindBin(xMin);
  Int_t histoBinMax = histo->FindBin(xMax);
  Int_t probBinMin = probHisto->FindBin(xMin);
  Int_t probBinMax = probHisto->FindBin(xMax);
  double histoIntegral = histo->Integral(histoBinMin, histoBinMax, "width");
  double probHistoIntegral = probHisto->Integral(probBinMin, probBinMax, "width");

  histo->Scale(1./histoIntegral);
  histo->Draw();

  probHisto->SetLineColor(kRed);
  probHisto->Scale(1./probHistoIntegral);
  probHisto->Draw("same");
}
