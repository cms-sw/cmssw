/**
 * This macro can be used to compare the BiasCheck_0 peak of a CB fit (produced by CompareBias.cc)
 * with the expected peak from the model convoluted with a gaussian.
 */
{
  inputFile1 = new TFile("BiasCheck_0.root", "READ");
  inputFile1->cd("MassVsEta");
  sigmaCanvas1 = (TCanvas*)gDirectory->Get("sigmaCanvas");
  sigmaHisto1 = (TH1D*)sigmaCanvas1->GetPrimitive("sigmaHisto");
  meanCanvas1 = (TCanvas*)gDirectory->Get("meanCanvas");
  meanHisto1 = (TH1D*)meanCanvas1->GetPrimitive("meanHisto");

  inputFile = new TFile("BiasCheck_3.root", "READ");
  inputFile->cd("MassVsEta");
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
      expectedMeanHisto->SetBinError(i, 0);
    }
  }

  newCanvas = new TCanvas("newCanvas", "newCanvas", 1000, 800);
  newCanvas->cd();
  newCanvas->Draw();
  meanHisto1->Draw();
  meanHisto->Draw("same");
  meanHisto->SetLineColor(2);
  expectedMeanHisto->SetMarkerColor(4);
  expectedMeanHisto->SetMarkerStyle(5);
  expectedMeanHisto->SetMarkerSize(5);
  expectedMeanHisto->SetLineColor(4);
  expectedMeanHisto->Draw("samep");
  // sigmaHisto->Draw();
  leg = new TLegend(0.1,0.7,0.48,0.9);
  leg->SetHeader("CrystalBall fit peak");
  leg->AddEntry(meanHisto1, "before correction", "l");
  leg->AddEntry(meanHisto, "after correction", "l");
  leg->AddEntry(expectedMeanHisto, "Expected from model", "p");
  leg->Draw("same");

  meanHisto1->GetXaxis()->SetRangeUser(-2.4, 2.4);
  meanHisto1->GetYaxis()->SetRangeUser(3.088, 3.099);
  meanHisto1->GetXaxis()->SetTitle("muon #eta");
  meanHisto1->GetYaxis()->SetTitle("peak (GeV)");
}
