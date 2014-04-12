/**
 * This macro can be used to compare the BiasCheck_0 peak of a CB fit (produced by CompareBias.cc)
 * with the expected peak from the model convoluted with a gaussian.
 */
{
  inputFile = new TFile("BiasCheck_0.root", "READ");
  inputFile->cd("MassVsEta");
  // gDirectory->ls();
  sigmaCanvas = (TCanvas*)gDirectory->Get("sigmaCanvas");
  sigmaHisto = (TH1D*)sigmaCanvas->GetPrimitive("sigmaHisto");

  meanCanvas = (TCanvas*)gDirectory->Get("meanCanvas");
  meanHisto = (TH1D*)meanCanvas->GetPrimitive("meanHisto");

  func = new TF1("linear", "[0] + [1]*x", 20., 82.);
  func->SetParameter(0, 3.09625);
  func->SetParameter(1, -0.0000380253);

  expectedMeanHisto = (TH1D*)meanHisto->Clone("expectedMeanHisto");

  std::cout << "NbinsX = " << meanHisto->GetNbinsX() << std::endl;
  for( int i=1; i<=meanHisto->GetNbinsX(); ++i ) {
    double sigma = sigmaHisto->GetBinContent(i);
    expectedMeanHisto->SetBinContent(i, 0);
    if( sigma != 0 ) {
      std::cout << "mean("<<i<<") = " << meanHisto->GetBinContent(i) << std::endl;
      // std::cout << "sigma("<<i<<") = " << sigma << std::endl;
      std::cout << "expected mean("<<i<<") = " << func->Eval(sigma*1000) << std::endl;
      expectedMeanHisto->SetBinContent(i, func->Eval(sigma*1000));
    }
  }

  newCanvas = new TCanvas("newCanvas", "newCanvas", 1000, 800);
  newCanvas->cd();
  newCanvas->Draw();
  meanHisto->Draw();
  expectedMeanHisto->SetMarkerColor(kRed);
  expectedMeanHisto->SetLineColor(kRed);
  expectedMeanHisto->Draw("same");
  // sigmaHisto->Draw();
  leg = new TLegend(0.1,0.7,0.48,0.9);
  leg->SetHeader("CrystalBall fit peak");
  leg->AddEntry(meanHisto, "Reconstructed MC", "l");
  leg->AddEntry(expectedMeanHisto, "Expected from model", "l");
  leg->Draw("same");

  meanHisto->GetXaxis()->SetRangeUser(-2.4, 2.4);
  meanHisto->GetYaxis()->SetRangeUser(3.092, 3.1);
  meanHisto->GetXaxis()->SetTitle("muon #eta");
  meanHisto->GetYaxis()->SetTitle("peak (GeV)");
}
