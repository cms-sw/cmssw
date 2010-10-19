#include <iostream>
#include <TH1D.h>
#include <TCanvas.h>
#include <TFile.h>
#include <TLegend.h>
#include <TGraphAsymmErrors.h>

void drawHisto(TString type, TFile * outputFile,
	       const double & minX, const double & maxX,
	       const double & minY, const double & maxY)
{
  TFile * inputFileData = new TFile("data.root", "READ");
  TCanvas * canvasData = (TCanvas*)inputFileData->Get("canvassigmaPtVs"+type);
  TGraphAsymmErrors * graphData = (TGraphAsymmErrors*)canvasData->GetPrimitive("Graph_from_sigmaPtVs"+type);


  TFile * inputFile = new TFile("test.root", "READ");
  TCanvas * canvas = (TCanvas*)inputFile->Get("canvassigmaPtVs"+type);
  TGraphAsymmErrors * graph = (TGraphAsymmErrors*)canvas->GetPrimitive("Graph_from_sigmaPtVs"+type);

  TFile * inputFile2 = new TFile("ComparedResol.root", "READ");
  TCanvas * canvas2 = (TCanvas*)inputFile2->Get("resolPtVS"+type);
  TH1D * histo = (TH1D*)canvas2->GetPrimitive("hResolPtGenVSMu_ResoVS"+type+"_resol_after");

//   Double_t x[n]   = {-0.22, 0.05, 0.25, 0.35, 0.5, 0.61,0.7,0.85,0.89,0.95};
//   Double_t y[n]   = {1,2.9,5.6,7.4,9,9.6,8.7,6.3,4.5,1};
//   Double_t exl[n] = {.05,.1,.07,.07,.04,.05,.06,.07,.08,.05};
//   Double_t eyl[n] = {.8,.7,.6,.5,.4,.4,.5,.6,.7,.8};
//   Double_t exh[n] = {.02,.08,.05,.05,.03,.03,.04,.05,.06,.03};
//   Double_t eyh[n] = {.6,.5,.4,.3,.2,.2,.3,.4,.5,.6};
//   gr = new TGraphAsymmErrors(n,x,y,exl,exh,eyl,eyh);

  int N = graph->GetN();
  Double_t * x = new Double_t[N];
  Double_t * y = new Double_t[N];
  Double_t * exl = new Double_t[N];
  Double_t * eyl = new Double_t[N];
  Double_t * exh = new Double_t[N];
  Double_t * eyh = new Double_t[N];

  Double_t * exlData = new Double_t[N];
  Double_t * eylData = new Double_t[N];
  Double_t * exhData = new Double_t[N];
  Double_t * eyhData = new Double_t[N];
  Double_t * xData = new Double_t[N];
  Double_t * yData = new Double_t[N];

  // Loop on the bins of the MC-truth histogram and compute the systematic error from the fit
  for( int i=0; i<N; ++i ) {
    double fitValue = graph->GetY()[i];
    double fitBinValue = graph->GetX()[i];
    double fitErrorPos = graph->GetEYhigh()[i];
    double fitErrorNeg = graph->GetEYlow()[i];
    int bin = histo->FindBin(fitBinValue);
    double binContent = histo->GetBinContent(bin);
    std::cout << "fitValue("<<i<<") = " << fitValue << " + " << fitErrorPos << " - " << fitErrorNeg << std::endl;
    std::cout << "binContent("<<i<<") = " << binContent << std::endl;
    double delta = fitValue - binContent;
    std::cout << "diff("<<i<<") = " << delta << std::endl;

    x[i] = fitBinValue;
    y[i] = fitValue;
    exl[i] = graph->GetEXlow()[i];
    eyl[i] = fitErrorNeg;
    exh[i] = graph->GetEXhigh()[i];
    eyh[i] = fitErrorPos;

    xData[i] = graphData->GetX()[i];
    yData[i] = graphData->GetY()[i];
    exlData[i] = graphData->GetEXlow()[i];
    eylData[i] = graphData->GetEYlow()[i];
    exhData[i] = graphData->GetEXhigh()[i];
    eyhData[i] = graphData->GetEYhigh()[i];

    if( delta > 0 ) {
      std::cout << "before eyl["<<i<<"] = " << eyl[i] << std::endl;
      eylData[i] = sqrt(fitErrorNeg*fitErrorNeg + delta*delta);
      std::cout << "after eyl["<<i<<"] = " << eyl[i] << std::endl;
    }
    else {
      std::cout << "before eyh["<<i<<"] = " << eyh[i] << std::endl;
      eyhData[i] = sqrt(fitErrorPos*fitErrorPos + delta*delta);
      std::cout << "after eyh["<<i<<"] = " << eyh[i] << std::endl;
    }
  }

  TGraphAsymmErrors * newGraph = new TGraphAsymmErrors(N,x,y,exl,exh,eyl,eyh);
  TGraphAsymmErrors * newGraphData = new TGraphAsymmErrors(N,xData,yData,exlData,exhData,eylData,eyhData);
  TGraph * lineGraph = new TGraph(N, xData, yData);
  newGraph->SetTitle("TGraphAsymmErrors Example");
  // TGraphAsymmErrors * newGraph = graph;

  TLegend * legend = new TLegend(0.7,0.71,0.98,1.);
  legend->SetTextSize(0.02);
  legend->SetFillColor(0); // Have a white background
  legend->AddEntry(histo, "resolution from fit on MC");
  legend->AddEntry(newGraphData, "resolution from fit on Data");

  newGraph->GetXaxis()->SetRangeUser(minX, maxX);
  newGraph->GetYaxis()->SetRangeUser(minY, maxY);
  // canvas->Draw();
  TCanvas * newCanvas = new TCanvas("newCanvas", "newCanvas", 1000, 800);
  newCanvas->Draw();
  newGraphData->SetFillColor(kGray);
  newGraphData->Draw("A2");
  newGraph->Draw("P");
  // histo->Draw("SAME");
  // graph->Draw("SAME");
  lineGraph->Draw("SAME");
  legend->Draw("SAME");

  outputFile->cd();
  canvas->Write();
}

void CompareErrorResol()
{
  TFile * outputFile = new TFile("output.root", "RECREATE");
  drawHisto("Pt", outputFile, 0., 20., 0., 0.06);
  drawHisto("Eta", outputFile, -2.39, 2.39, 0., 0.06);
  outputFile->Write();
  outputFile->Close();
}
