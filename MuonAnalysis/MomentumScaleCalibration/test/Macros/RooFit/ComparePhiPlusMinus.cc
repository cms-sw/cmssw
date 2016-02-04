{
  inputFile1 = new TFile("BiasCheck_0.root");
  // inputFile2 = new TFile("BiasCheck_0.root");

  inputFile1->cd("MassVsPhiMinus");
  canvas1 = (TCanvas*)gDirectory->Get("meanCanvas");
  histo1 = (TH1D*)canvas1->GetPrimitive("meanHisto");

  inputFile1->cd("MassVsPhiPlus");
  canvas2 = (TCanvas*)gDirectory->Get("meanCanvas");
  histo2 = (TH1D*)canvas2->GetPrimitive("meanHisto");

  TCanvas * newCanvas = new TCanvas("newCanvas", "newCanvas", 1000, 800);
  newCanvas->cd();
  newCanvas->Draw();
  histo1->Draw();
  histo2->Draw("same");
  histo2->SetLineColor(kRed);
}
