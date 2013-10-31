
#include <TFile.h>
#include <TChain.h>
#include <TTree.h>
#include <TTreeFormula.h>
#include <TString.h>
#include <TObjArray.h>
#include <TObjString.h>
#include <TH1.h>
#include <TH2.h>
#include <TGraphAsymmErrors.h>
#include <TGraphErrors.h>
#include <TF1.h>
#include <TFormula.h>
#include <TPaveText.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TMath.h>
#include <TROOT.h>
#include <TSystem.h>

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <iomanip>
#include <assert.h>
#include <math.h>
#include <limits>

enum { kTauPt, kLogTauPt };
enum { kTauEta, kTauAbsEta };

void normalizeHistogram(TH1* histogram)
{
  if ( histogram->Integral() > 0. ) {
    if ( !histogram->GetSumw2N() ) histogram->Sumw2();
    histogram->Scale(1./histogram->Integral());
  } 
}

//-------------------------------------------------------------------------------
void getBinomialBounds(Int_t n, Int_t r, Float_t& rMin, Float_t& rMax)
{
  rMin = 0.;
  rMax = 0.;

  if ( n == 0 ){
    return;
  }
  if ( r < 0 ){
    std::cerr << "Error in <getBinomialBounds>: n = " << n << ", r = " << r << std::endl;
    return;
  }
  
  if ( ((Double_t)r*(n - r)) > (9.*n) ){
    rMin = r - TMath::Sqrt((Double_t)r*(n - r)/((Double_t)n));
    rMax = r + TMath::Sqrt((Double_t)r*(n - r)/((Double_t)n));
    return;
  }

  Double_t binomialCoefficient = 1.;

  Double_t rMinLeft       = 0.;
  Double_t rMinMiddle     = TMath::Max(0.5*r, n - 1.5*r);
  Double_t rMinRight      = n;
  Double_t rMinLeftProb   = 0.;
  Double_t rMinMiddleProb = 0.5;
  Double_t rMinRightProb  = 1.;
  while ( (rMinRight - rMinLeft) > (0.001*n) ){

    rMinMiddleProb = 0;
    for ( Int_t i = r; i <= n; i++ ){
      binomialCoefficient = 1;

      for ( Int_t j = n; j > i; j-- ){
        binomialCoefficient *= j/((Double_t)(j - i));
      }

      rMinMiddleProb += binomialCoefficient*TMath::Power(rMinMiddle/((Double_t)(n)), i)
                       *TMath::Power((n - rMinMiddle)/((Double_t)(n)), n - i);
    }

    if ( rMinMiddleProb > 0.16 ){
      rMinRight     = rMinMiddle;
      rMinRightProb = rMinMiddleProb;
    } else if ( rMinMiddleProb < 0.16 ){
      rMinLeft      = rMinMiddle;
      rMinLeftProb  = rMinMiddleProb;
    } else {
      rMinLeft      = rMinRight     = rMinMiddle;
      rMinLeftProb  = rMinRightProb = rMinMiddleProb;
    }

    rMinMiddle = 0.5*(rMinLeft + rMinRight);

    if ( rMinLeft > r ){
      rMinMiddle = rMinLeft = rMinRight = 0;
    }
  }

  Double_t rMaxLeft       = 0.;
  Double_t rMaxMiddle     = TMath::Min(1.5*r, n - 0.5*r);
  Double_t rMaxRight      = n;
  Double_t rMaxLeftProb   = 1.;
  Double_t rMaxMiddleProb = 0.5;
  Double_t rMaxRightProb  = 0.;
  while ( (rMaxRight - rMaxLeft) > (0.001*n) ){

    rMaxMiddleProb = 0;
    for ( Int_t i = 0; i <= r; i++ ){
      binomialCoefficient = 1;
      
      for ( Int_t j = n; j > (n - i); j-- ){
        binomialCoefficient *= j/((Double_t)(i - (n - j)));
      }

      rMaxMiddleProb += binomialCoefficient*TMath::Power(rMaxMiddle/((Double_t)(n)), i)
                       *TMath::Power((n - rMaxMiddle)/((Double_t)(n)), n - i);
    }

    if ( rMaxMiddleProb > 0.16 ){
      rMaxLeft      = rMaxMiddle;
      rMaxLeftProb  = rMaxMiddleProb;
    } else if ( rMaxMiddleProb < 0.16 ){
      rMaxRight     = rMaxMiddle;
      rMaxRightProb = rMaxMiddleProb;
    } else {
      rMaxLeft      = rMaxRight     = rMaxMiddle;
      rMaxLeftProb  = rMaxRightProb = rMaxMiddleProb;
    }

    rMaxMiddle = 0.5*(rMaxLeft + rMaxRight);

    if ( rMaxRight < r ){
      rMaxMiddle = rMaxLeft = rMaxRight = n;
    }
  }

  rMin = rMinMiddle;
  rMax = rMaxMiddle;
}

TGraphAsymmErrors* getEfficiency(const TH1* histogram_numerator, const TH1* histogram_denominator)
{
  Int_t error = 0;
  if ( !(histogram_numerator->GetNbinsX()           == histogram_denominator->GetNbinsX())           ) error = 1;
  if ( !(histogram_numerator->GetXaxis()->GetXmin() == histogram_denominator->GetXaxis()->GetXmin()) ) error = 1;
  if ( !(histogram_numerator->GetXaxis()->GetXmax() == histogram_denominator->GetXaxis()->GetXmax()) ) error = 1;
  
  if ( error ){
    std::cerr << "Error in <getEfficiency>: Dimensionality of histograms does not match !!" << std::endl;
    return 0;
  }
  
  TAxis* xAxis = histogram_numerator->GetXaxis();

  Int_t nBins = xAxis->GetNbins();
  TArrayF x(nBins);
  TArrayF dxUp(nBins);
  TArrayF dxDown(nBins);
  TArrayF y(nBins);
  TArrayF dyUp(nBins);
  TArrayF dyDown(nBins);

  for ( Int_t ibin = 1; ibin <= nBins; ibin++ ){
    Int_t nObs = TMath::Nint(histogram_denominator->GetBinContent(ibin));
    Int_t rObs = TMath::Nint(histogram_numerator->GetBinContent(ibin));

    Float_t xCenter = histogram_denominator->GetBinCenter(ibin);
    Float_t xWidth  = histogram_denominator->GetBinWidth(ibin);

    x[ibin - 1]      = xCenter;
    dxUp[ibin - 1]   = 0.5*xWidth;
    dxDown[ibin - 1] = 0.5*xWidth;
    
    if ( nObs > 0 ){
      Float_t rMin = 0.;
      Float_t rMax = 0.;
      
      getBinomialBounds(nObs, rObs, rMin, rMax);

      y[ibin - 1]      = rObs/((Float_t)nObs);
      dyUp[ibin - 1]   = (rMax - rObs)/((Float_t)nObs);
      dyDown[ibin - 1] = (rObs - rMin)/((Float_t)nObs);
    } else{
      y[ibin - 1]      = 0.;
      dyUp[ibin - 1]   = 0.;
      dyDown[ibin - 1] = 0.;
    }
  }
  
  TString name  = TString(histogram_numerator->GetName()).Append("Graph");
  TString title = histogram_numerator->GetTitle();

  TGraphAsymmErrors* graph = 
    new TGraphAsymmErrors(nBins, x.GetArray(), y.GetArray(), 
			  dxDown.GetArray(), dxUp.GetArray(), dyDown.GetArray(), dyUp.GetArray());

  graph->SetName(name);
  graph->SetTitle(title);

  return graph;
}

void showEfficiency(const TString& title, double canvasSizeX, double canvasSizeY,
		    const TH1* histogram1_numerator, const TH1* histogram1_denominator, const std::string& legendEntry1,
		    const TH1* histogram2_numerator, const TH1* histogram2_denominator, const std::string& legendEntry2,
		    const TH1* histogram3_numerator, const TH1* histogram3_denominator, const std::string& legendEntry3,
		    const TH1* histogram4_numerator, const TH1* histogram4_denominator, const std::string& legendEntry4,
		    const TH1* histogram5_numerator, const TH1* histogram5_denominator, const std::string& legendEntry5,
		    const TH1* histogram6_numerator, const TH1* histogram6_denominator, const std::string& legendEntry6,
		    const std::string& xAxisTitle, double xAxisOffset,
                    bool useLogScale, double yMin, double yMax, const std::string& yAxisTitle, double yAxisOffset,
		    double legendX0, double legendY0, 
		    const std::string& outputFileName)
{
  TCanvas* canvas = new TCanvas("canvas", "canvas", canvasSizeX, canvasSizeY);
  canvas->SetFillColor(10);
  canvas->SetBorderSize(2);
  canvas->SetLeftMargin(0.12);
  canvas->SetBottomMargin(0.12);
  canvas->SetLogy(useLogScale);
  canvas->SetGridx();
  canvas->SetGridy();

  TH1* dummyHistogram = new TH1D("dummyHistogram_top", "dummyHistogram_top", 10, histogram1_numerator->GetXaxis()->GetXmin(), histogram1_numerator->GetXaxis()->GetXmax());
  dummyHistogram->SetTitle("");
  dummyHistogram->SetStats(false);
  dummyHistogram->SetMaximum(yMax);
  dummyHistogram->SetMinimum(yMin);
  
  TAxis* xAxis = dummyHistogram->GetXaxis();
  xAxis->SetTitle(xAxisTitle.data());
  xAxis->SetTitleOffset(xAxisOffset);
  
  TAxis* yAxis = dummyHistogram->GetYaxis();
  yAxis->SetTitle(yAxisTitle.data());
  yAxis->SetTitleOffset(yAxisOffset);

  dummyHistogram->Draw();

  int colors[6] = { 1, 2, 3, 4, 6, 7 };
  int markerStyles[6] = { 22, 32, 20, 24, 21, 25 };

  int numGraphs = 1;
  if ( histogram2_numerator && histogram2_denominator ) ++numGraphs;
  if ( histogram3_numerator && histogram3_denominator ) ++numGraphs;
  if ( histogram4_numerator && histogram4_denominator ) ++numGraphs;
  if ( histogram5_numerator && histogram5_denominator ) ++numGraphs;
  if ( histogram6_numerator && histogram6_denominator ) ++numGraphs;

  TLegend* legend = new TLegend(legendX0, legendY0, legendX0 + 0.18, legendY0 + 0.05*numGraphs, "", "brNDC"); 
  legend->SetBorderSize(0);
  legend->SetFillColor(0);
  
  TGraphAsymmErrors* graph1 = getEfficiency(histogram1_numerator, histogram1_denominator);
  graph1->SetLineColor(colors[0]);
  graph1->SetMarkerColor(colors[0]);
  graph1->SetMarkerStyle(markerStyles[0]);
  graph1->Draw("p");
  legend->AddEntry(graph1, legendEntry1.data(), "p");    

  TGraphAsymmErrors* graph2 = 0;
  if ( histogram2_numerator && histogram2_denominator ) {
    graph2 = getEfficiency(histogram2_numerator, histogram2_denominator);
    graph2->SetLineColor(colors[1]);
    graph2->SetMarkerColor(colors[1]);
    graph2->SetMarkerStyle(markerStyles[1]);
    graph2->Draw("p");
    legend->AddEntry(graph2, legendEntry2.data(), "p");
  }

  TGraphAsymmErrors* graph3 = 0;
  if ( histogram3_numerator && histogram3_denominator ) {
    graph3 = getEfficiency(histogram3_numerator, histogram3_denominator);
    graph3->SetLineColor(colors[2]);
    graph3->SetMarkerColor(colors[2]);
    graph3->SetMarkerStyle(markerStyles[2]);
    graph3->Draw("p");
    legend->AddEntry(graph3, legendEntry3.data(), "p");
  }
  
  TGraphAsymmErrors* graph4 = 0;
  if ( histogram4_numerator && histogram4_denominator ) {
    graph4 = getEfficiency(histogram4_numerator, histogram4_denominator);
    graph4->SetLineColor(colors[3]);
    graph4->SetMarkerColor(colors[3]);
    graph4->SetMarkerStyle(markerStyles[3]);
    graph4->Draw("p");
    legend->AddEntry(graph4, legendEntry4.data(), "p");
  }

  TGraphAsymmErrors* graph5 = 0;
  if ( histogram5_numerator && histogram5_denominator ) {
    graph5 = getEfficiency(histogram5_numerator, histogram5_denominator);
    graph5->SetLineColor(colors[4]);
    graph5->SetMarkerColor(colors[4]);
    graph5->SetMarkerStyle(markerStyles[4]);
    graph5->Draw("p");
    legend->AddEntry(graph5, legendEntry5.data(), "p");
  }
  
  TGraphAsymmErrors* graph6 = 0;
  if ( histogram6_numerator && histogram6_denominator ) {
    graph6 = getEfficiency(histogram6_numerator, histogram6_denominator);
    graph6->SetLineColor(colors[5]);
    graph6->SetMarkerColor(colors[5]);
    graph6->SetMarkerStyle(markerStyles[5]);
    graph6->Draw("p");
    legend->AddEntry(graph6, legendEntry6.data(), "p");
  }

  legend->Draw();

  TPaveText* label = 0;
  if ( title.Length() > 0 ) {
    label = new TPaveText(0.175, 0.925, 0.48, 0.98, "NDC");
    label->AddText(title.Data());
    label->SetTextAlign(13);
    label->SetTextSize(0.045);
    label->SetFillStyle(0);
    label->SetBorderSize(0);
    label->Draw();
  }

  canvas->Update();
  size_t idx = outputFileName.find_last_of('.');
  std::string outputFileName_plot = std::string(outputFileName, 0, idx);
  if ( idx != std::string::npos ) canvas->Print(std::string(outputFileName_plot).append(std::string(outputFileName, idx)).data());
  canvas->Print(std::string(outputFileName_plot).append(".png").data());
  canvas->Print(std::string(outputFileName_plot).append(".pdf").data());
  
  delete legend;
  delete label;
  delete dummyHistogram;
  delete canvas;
}
//-------------------------------------------------------------------------------

//-------------------------------------------------------------------------------
void showDistribution(const TString& title, double canvasSizeX, double canvasSizeY,
		      TH1* histogram1, const std::string& legendEntry1,
		      TH1* histogram2, const std::string& legendEntry2,
		      TH1* histogram3, const std::string& legendEntry3,
		      TH1* histogram4, const std::string& legendEntry4,
		      const std::string& xAxisTitle, double xAxisOffset,
		      bool useLogScale, double yMin, double yMax, const std::string& yAxisTitle, double yAxisOffset,
		      double legendX0, double legendY0, 
		      const std::string& outputFileName)
{
  TCanvas* canvas = new TCanvas("canvas", "canvas", canvasSizeX, canvasSizeY);
  canvas->SetFillColor(10);
  canvas->SetBorderSize(2);
  canvas->SetLeftMargin(0.12);
  canvas->SetBottomMargin(0.12);
  canvas->SetLogy(useLogScale);

  int colors[6] = { 1, 2, 3, 4, 6, 7 };
  int markerStyles[6] = { 22, 32, 20, 24, 21, 25 };

  int numHistograms = 1;
  if ( histogram2 ) ++numHistograms;
  if ( histogram3 ) ++numHistograms;
  if ( histogram4 ) ++numHistograms;

  TLegend* legend = new TLegend(legendX0, legendY0, legendX0 + 0.44, legendY0 + 0.05*numHistograms, "", "brNDC"); 
  legend->SetBorderSize(0);
  legend->SetFillColor(0);

  histogram1->SetTitle("");
  histogram1->SetStats(false);
  histogram1->SetMinimum(yMin);
  histogram1->SetMaximum(yMax);
  histogram1->SetLineColor(colors[0]);
  histogram1->SetLineWidth(2);
  histogram1->SetMarkerColor(colors[0]);
  histogram1->SetMarkerStyle(markerStyles[0]);
  histogram1->Draw("e1p");
  legend->AddEntry(histogram1, legendEntry1.data(), "p");

  TAxis* xAxis = histogram1->GetXaxis();
  xAxis->SetTitle(xAxisTitle.data());
  xAxis->SetTitleOffset(xAxisOffset);

  TAxis* yAxis = histogram1->GetYaxis();
  yAxis->SetTitle(yAxisTitle.data());
  yAxis->SetTitleOffset(yAxisOffset);

  if ( histogram2 ) {
    histogram2->SetLineColor(colors[1]);
    histogram2->SetLineWidth(2);
    histogram2->SetMarkerColor(colors[1]);
    histogram2->SetMarkerStyle(markerStyles[1]);
    histogram2->Draw("e1psame");
    legend->AddEntry(histogram2, legendEntry2.data(), "p");
  }

  if ( histogram3 ) {
    histogram3->SetLineColor(colors[2]);
    histogram3->SetLineWidth(2);
    histogram3->SetMarkerColor(colors[2]);
    histogram3->SetMarkerStyle(markerStyles[2]);
    histogram3->Draw("e1psame");
    legend->AddEntry(histogram3, legendEntry3.data(), "p");
  }

  if ( histogram4 ) {
    histogram4->SetLineColor(colors[3]);
    histogram4->SetLineWidth(2);
    histogram4->SetMarkerColor(colors[3]);
    histogram4->SetMarkerStyle(markerStyles[3]);
    histogram4->Draw("e1psame");
    legend->AddEntry(histogram4, legendEntry4.data(), "p");
  }

  legend->Draw();

  TPaveText* label = 0;
  if ( title.Length() > 0 ) {
    label = new TPaveText(0.175, 0.925, 0.48, 0.98, "NDC");
    label->AddText(title.Data());
    label->SetTextAlign(13);
    label->SetTextSize(0.045);
    label->SetFillStyle(0);
    label->SetBorderSize(0);
    label->Draw();
  }
  
  canvas->Update();
  size_t idx = outputFileName.find_last_of('.');
  std::string outputFileName_plot = std::string(outputFileName, 0, idx);
  if ( idx != std::string::npos ) canvas->Print(std::string(outputFileName_plot).append(std::string(outputFileName, idx)).data());
  canvas->Print(std::string(outputFileName_plot).append(".png").data());
  canvas->Print(std::string(outputFileName_plot).append(".pdf").data());
  
  delete legend;
  delete label;
  delete canvas;  
}
//-------------------------------------------------------------------------------

//-------------------------------------------------------------------------------
TGraph* compMVAcut(const TH2* histogramMVAoutput_vs_Pt, const TH1* histogramPt, double Efficiency_or_FakeRate)
{
  const TAxis* xAxis = histogramMVAoutput_vs_Pt->GetXaxis();
  int numBinsX = xAxis->GetNbins();

  const TAxis* yAxis = histogramMVAoutput_vs_Pt->GetYaxis();
  int numBinsY = yAxis->GetNbins();

  TGraph* graph = new TGraphAsymmErrors(numBinsX);
  std::string graphName = Form("%s_graph", histogramMVAoutput_vs_Pt->GetName());
  graph->SetName(graphName.data());

  int numPoints = 0;

  for ( int iBinX = 1; iBinX <= numBinsX; ++iBinX ) {
    double ptMin = xAxis->GetBinLowEdge(iBinX);
    double ptMax = xAxis->GetBinUpEdge(iBinX);

    int binLowIndex = const_cast<TH1*>(histogramPt)->FindBin(ptMin);
    int binUpIndex  = const_cast<TH1*>(histogramPt)->FindBin(ptMax);
    //std::cout << "ptMin = " << ptMin << ", ptMax = " << ptMax << ": binLowIndex = " << binLowIndex << ", binUpIndex = " << binUpIndex << std::endl;
    histogramPt->GetXaxis()->SetRange(binLowIndex, binUpIndex);
    // CV: skip bins of low statistics
    if ( histogramPt->GetEntries() < 100 ) {
      std::cout << "Warning: bin @ x = " << xAxis->GetBinCenter(iBinX) << " has low statistics (#entries = " << histogramPt->GetEntries() << ") --> skipping !!" << std::endl;
      continue;
    }

    double x = histogramPt->GetMean();

    //std::cout << "iBinX = " << iBinX << ": xMean = " << x << std::endl;

    double normalization = 0.;
    for ( int iBinY = numBinsY; iBinY >= 1; --iBinY ) {
      normalization += histogramMVAoutput_vs_Pt->GetBinContent(iBinX, iBinY);
    }
    // CV: skip bins of low statistics
    if ( normalization < 100 ) {
      std::cout << "Warning: bin @ x = " << xAxis->GetBinCenter(iBinX) << " has low statistics (normalization = " << normalization << ") --> skipping !!" << std::endl;
      continue;
    }
    
    double y = -1.;

    double runningSum = 0.;
    for ( int iBinY = numBinsY; iBinY >= 1; --iBinY ) {
      double binContent_normalized = histogramMVAoutput_vs_Pt->GetBinContent(iBinX, iBinY)/normalization;
      if ( (runningSum + binContent_normalized) > Efficiency_or_FakeRate ) {
	y = yAxis->GetBinUpEdge(iBinY) - ((Efficiency_or_FakeRate - runningSum)/binContent_normalized)*yAxis->GetBinWidth(iBinY);
	//std::cout << "iBinY = " << iBinY << " (yCenter = " << yAxis->GetBinCenter(iBinY) << "): binContent = " << binContent_normalized << std::endl;
	//std::cout << "--> setting y = " << y << std::endl;
	assert(y >= yAxis->GetBinLowEdge(iBinY));
	break;
      } else {
	runningSum += binContent_normalized;
	//std::cout << "iBinY = " << iBinY << " (yCenter = " << yAxis->GetBinCenter(iBinY) << "): runningSum = " << runningSum << std::endl;
      }
    }
    
    graph->SetPoint(numPoints, x, y);
    ++numPoints;
  }

  for ( int iPoint = numBinsX; iPoint > numPoints; --iPoint ) {
    //std::cout << "removing point #" << iPoint << std::endl;
    graph->RemovePoint(graph->GetN() - 1);
  }

  // reset x-axis range selection 
  histogramPt->GetXaxis()->SetRange(1., 0.);

  return graph;
}

void showGraphs(const TString& title, double canvasSizeX, double canvasSizeY,
		TGraph* graph1, const std::string& legendEntry1,
		TGraph* graph2, const std::string& legendEntry2,
		TGraph* graph3, const std::string& legendEntry3,
		TGraph* graph4, const std::string& legendEntry4,
		TGraph* graph5, const std::string& legendEntry5,
		TGraph* graph6, const std::string& legendEntry6,
		double xMin, double xMax, unsigned numBinsX, const std::string& xAxisTitle, double xAxisOffset,
		double yMin, double yMax, const std::string& yAxisTitle, double yAxisOffset,
		double legendX0, double legendY0, 
		const std::string& outputFileName)
{
  TCanvas* canvas = new TCanvas("canvas", "canvas", canvasSizeX, canvasSizeY);
  canvas->SetFillColor(10);
  canvas->SetBorderSize(2);
  canvas->SetLeftMargin(0.12);
  canvas->SetBottomMargin(0.12);

  int colors[6] = { 1, 2, 3, 4, 6, 7 };
  int markerStyles[6] = { 22, 32, 20, 24, 21, 25 };

  TLegend* legend = new TLegend(legendX0, legendY0, legendX0 + 0.44, legendY0 + 0.20, "", "brNDC"); 
  legend->SetBorderSize(0);
  legend->SetFillColor(0);

  TH1* dummyHistogram = new TH1D("dummyHistogram", "dummyHistogram", numBinsX, xMin, xMax);
  dummyHistogram->SetTitle("");
  dummyHistogram->SetStats(false);
  dummyHistogram->SetMinimum(yMin);
  dummyHistogram->SetMaximum(yMax);

  TAxis* xAxis = dummyHistogram->GetXaxis();
  xAxis->SetTitle(xAxisTitle.data());
  xAxis->SetTitleOffset(xAxisOffset);

  TAxis* yAxis = dummyHistogram->GetYaxis();
  yAxis->SetTitle(yAxisTitle.data());
  yAxis->SetTitleOffset(yAxisOffset);

  dummyHistogram->Draw("axis");

  graph1->SetLineColor(colors[0]);
  graph1->SetLineWidth(2);
  graph1->Draw("L");
  legend->AddEntry(graph1, legendEntry1.data(), "l");

  if ( graph2 ) {
    graph2->SetLineColor(colors[1]);
    graph2->SetLineWidth(2);
    graph2->Draw("L");
    legend->AddEntry(graph2, legendEntry2.data(), "l");
  }
  
  if ( graph3 ) {
    graph3->SetLineColor(colors[2]);
    graph3->SetLineWidth(2);
    graph3->Draw("L");
    legend->AddEntry(graph3, legendEntry3.data(), "l");
  }

  if ( graph4 ) {
    graph4->SetLineColor(colors[3]);
    graph4->SetLineWidth(2);
    graph4->Draw("L");
    legend->AddEntry(graph4, legendEntry4.data(), "l");
  }

  if ( graph5 ) {
    graph5->SetLineColor(colors[4]);
    graph5->SetLineWidth(2);
    graph5->Draw("L");
    legend->AddEntry(graph5, legendEntry5.data(), "l");
  }

  if ( graph6 ) {
    graph6->SetLineColor(colors[5]);
    graph6->SetLineWidth(2);
    graph6->Draw("L");
    legend->AddEntry(graph6, legendEntry6.data(), "l");
  }
  
  legend->Draw();
    
  TPaveText* label = 0;
  if ( title.Length() > 0 ) {
    label = new TPaveText(0.175, 0.925, 0.48, 0.98, "NDC");
    label->AddText(title.Data());
    label->SetTextAlign(13);
    label->SetTextSize(0.045);
    label->SetFillStyle(0);
    label->SetBorderSize(0);
    label->Draw();
  }

  canvas->Update();
  size_t idx = outputFileName.find_last_of('.');
  std::string outputFileName_plot = std::string(outputFileName, 0, idx);
  if ( idx != std::string::npos ) canvas->Print(std::string(outputFileName_plot).append(std::string(outputFileName, idx)).data());
  canvas->Print(std::string(outputFileName_plot).append(".png").data());
  canvas->Print(std::string(outputFileName_plot).append(".pdf").data());
  
  delete legend;
  delete label;
  delete dummyHistogram;
  delete canvas;  
}
//-------------------------------------------------------------------------------

struct plotEntryType
{
  plotEntryType(const std::string& name, double mvaCut)
    : name_(name),
      mvaCut_(mvaCut),
      histogramPt_numerator_(0),
      histogramPt_denominator_(0),
      histogramEta_numerator_(0),
      histogramEta_denominator_(0),
      histogramNvtx_numerator_(0),
      histogramNvtx_denominator_(0),
      histogramMVAoutput_vs_Pt_(0),
      histogramPt_(0)
  {}
  ~plotEntryType()
  {
    delete histogramPt_numerator_;
    delete histogramPt_denominator_;
    delete histogramEta_numerator_;
    delete histogramEta_denominator_;
    delete histogramNvtx_numerator_;
    delete histogramNvtx_denominator_;
    delete histogramMVAoutput_vs_Pt_;
    delete histogramPt_;
  }
  void bookHistograms()
  {
    const int ptNumBins = 26;
    double ptBinning[ptNumBins + 1] = { 
      20., 22.5, 25., 27.5, 30., 32.5, 35., 37.5, 40., 45., 50., 55., 60., 70., 80., 90., 100., 125., 150., 175., 200., 250., 300., 400., 500., 1000., 5000.
    };  
    std::string histogramNamePt_numerator = Form("histogramPt_%s_numerator", name_.data());
    histogramPt_numerator_ = new TH1D(histogramNamePt_numerator.data(), histogramNamePt_numerator.data(), ptNumBins, ptBinning);
    std::string histogramNamePt_denominator = Form("histogramPt_%s_denominator", name_.data());
    histogramPt_denominator_ = new TH1D(histogramNamePt_denominator.data(), histogramNamePt_denominator.data(), ptNumBins, ptBinning);
    std::string histogramNameEta_numerator = Form("histogramEta_%s_numerator", name_.data());
    histogramEta_numerator_ = new TH1D(histogramNameEta_numerator.data(), histogramNameEta_numerator.data(), 23, 0., 2.3);
    std::string histogramNameEta_denominator = Form("histogramEta_%s_denominator", name_.data());
    histogramEta_denominator_ = new TH1D(histogramNameEta_denominator.data(), histogramNameEta_denominator.data(), 23, 0., 2.3);
    std::string histogramNameNvtx_numerator = Form("histogramNvtx_%s_numerator", name_.data());
    histogramNvtx_numerator_ = new TH1D(histogramNameNvtx_numerator.data(), histogramNameNvtx_numerator.data(), 60, -0.5, 59.5);
    std::string histogramNameNvtx_denominator = Form("histogramNvtx_%s_denominator", name_.data());
    histogramNvtx_denominator_ = new TH1D(histogramNameNvtx_denominator.data(), histogramNameNvtx_denominator.data(), 60, -0.5, 59.5);
    std::string histogramNameMVAoutput_vs_Pt = Form("histogramMVAoutput_vs_Pt_%s", name_.data());
    histogramMVAoutput_vs_Pt_ = new TH2D(histogramNameMVAoutput_vs_Pt.data(), histogramNameMVAoutput_vs_Pt.data(), ptNumBins, ptBinning, 20200, -1.01, +1.01);
    std::string histogramNamePt = Form("histogramPt_%s", name_.data());
    histogramPt_ = new TH1D(histogramNamePt.data(), histogramNamePt.data(), 2500, 0., 2500.);
  }
  void fillHistograms(double mvaOutput, double pt, double eta, double Nvtx, double evtWeight)
  {    
    histogramPt_denominator_->Fill(pt, evtWeight);
    histogramEta_denominator_->Fill(eta, evtWeight);
    histogramNvtx_denominator_->Fill(Nvtx, evtWeight);
    bool passesCuts = (mvaOutput > mvaCut_);
    //std::cout << "passesCuts = " << passesCuts << std::endl;
    if ( passesCuts ) {
      histogramPt_numerator_->Fill(pt, evtWeight);
      histogramEta_numerator_->Fill(eta, evtWeight);
      histogramNvtx_numerator_->Fill(Nvtx, evtWeight);
    }
    double y = mvaOutput;
    TAxis* yAxis = histogramMVAoutput_vs_Pt_->GetYaxis();
    int binY = yAxis->FindBin(y);
    int numBinsY = yAxis->GetNbins();
    if ( binY <  1       ) binY = 1;
    if ( binY > numBinsY ) binY = numBinsY;
    double yWithinRange = yAxis->GetBinCenter(binY);
    histogramMVAoutput_vs_Pt_->Fill(pt, y, evtWeight);
    histogramPt_->Fill(pt, evtWeight);
  }
  std::string name_;
  double mvaCut_;
  TH1* histogramPt_numerator_;
  TH1* histogramPt_denominator_;
  TH1* histogramEta_numerator_;
  TH1* histogramEta_denominator_;
  TH1* histogramNvtx_numerator_;
  TH1* histogramNvtx_denominator_;
  TH2* histogramMVAoutput_vs_Pt_;
  TH1* histogramPt_;
};

void fillPlots(const std::string& inputFileName, plotEntryType* plots_signal, plotEntryType* plots_background, 
	       double mvaCut, 
	       int tauPtMode, int tauEtaMode,
	       TFormula* mvaOutput_normalization)
{
  std::string treeName = "TrainTree";
  
  int classId_signal     = 0;
  int classId_background = 1;

  TFile* inputFile = new TFile(inputFileName.data());

  TTree* tree = dynamic_cast<TTree*>(inputFile->Get(treeName.data()));

  Float_t recTauPt, recLogTauPt, recTauEta, recTauAbsEta;
  if ( tauPtMode == kTauPt ) {
    tree->SetBranchAddress("recTauPt", &recTauPt);
  } else if ( tauPtMode == kLogTauPt ) {
    tree->SetBranchAddress("TMath_Log_TMath_Max_1.,recTauPt__", &recLogTauPt);
  } else assert(0);
  if ( tauEtaMode == kTauEta ) {
    tree->SetBranchAddress("recTauEta", &recTauEta);
  } else if ( tauEtaMode == kTauAbsEta ) {
    tree->SetBranchAddress("TMath_Abs_recTauEta_", &recTauAbsEta);
  } else assert(0);

  Float_t numVertices;
  tree->SetBranchAddress("numOfflinePrimaryVertices", &numVertices);

  Float_t mvaOutput;
  tree->SetBranchAddress("BDTG", &mvaOutput);

  Int_t classId;
  tree->SetBranchAddress("classID", &classId);
  
  Float_t evtWeight;
  //tree->SetBranchAddress("weight", &evtWeight);
  evtWeight = 1.0;
  
  const int maxEvents = -1;
  //const int maxEvents = 1000000;

  int numEntries = tree->GetEntries();

  double normalization_signal = 0.;
  double normalization_background = 0.;
  for ( int iEntry = 0; iEntry < numEntries && (iEntry < maxEvents || maxEvents == -1); ++iEntry ) {
    if ( iEntry > 0 && (iEntry % 10000) == 0 ) {
      std::cout << "processing Entry " << iEntry << std::endl;
    }
    
    tree->GetEntry(iEntry);
    //std::cout << "Entry #" << iEntry << " (classId = " << classId << "): evtWeight = " << evtWeight << std::endl;

    if ( classId == classId_signal ) normalization_signal += evtWeight;
    else if ( classId == classId_background ) normalization_background += evtWeight;
  }
  std::cout << "normalization: signal = " << normalization_signal << ", background = " << normalization_background << std::endl;  

  mvaOutput_normalization->SetParameter(0, normalization_signal);
  mvaOutput_normalization->SetParameter(1, normalization_background);

  for ( int iEntry = 0; iEntry < numEntries && (iEntry < maxEvents || maxEvents == -1); ++iEntry ) {
    if ( iEntry > 0 && (iEntry % 10000) == 0 ) {
      std::cout << "processing Entry " << iEntry << std::endl;
    }
    
    tree->GetEntry(iEntry);

    if ( tauPtMode == kLogTauPt ) {
      recTauPt = TMath::Exp(recLogTauPt);
    }
    if ( tauEtaMode == kTauEta ) {
      recTauAbsEta = TMath::Abs(recTauEta);
    }

    //double mvaOutput_normalized = 1. - (1. - mvaOutput)*(normalization_signal/normalization_background);
    double mvaOutput_normalized = mvaOutput_normalization->Eval(mvaOutput);
    if ( mvaOutput_normalized > 1. ) mvaOutput_normalized = 1.;
    if ( mvaOutput_normalized < 0. ) mvaOutput_normalized = 0.;
    //double mvaOutput_normalized = mvaOutput;
    //if ( mvaOutput_normalized > +1. ) mvaOutput_normalized = +1.;
    //if ( mvaOutput_normalized < -1. ) mvaOutput_normalized = -1.;
    //std::cout << "Entry #" << iEntry << " (classId = " << classId << "): MVA output = " << mvaOutput << " --> normalized = " << mvaOutput_normalized << std::endl;

    if ( classId == classId_signal ) plots_signal->fillHistograms(mvaOutput_normalized, recTauPt, recTauAbsEta, numVertices, evtWeight);
    else if ( classId == classId_background ) plots_background->fillHistograms(mvaOutput_normalized, recTauPt, recTauAbsEta, numVertices, evtWeight);
  }

  delete tree;

  delete inputFile;
}

struct mvaEntryType
{
  mvaEntryType(const std::string& inputFileName, const std::string& mvaName, double mvaCut, int tauPtMode, int tauEtaMode)
    : inputFileName_(inputFileName),
      mvaName_(mvaName),
      mvaCut_(mvaCut),
      tauPtMode_(tauPtMode),
      tauEtaMode_(tauEtaMode)      
  {
    legendEntry_ = Form("MVA %s", mvaName.data());
    plots_signal_ = new plotEntryType("signal", mvaCut);
    plots_signal_->bookHistograms();
    plots_background_ = new plotEntryType("background", mvaCut);
    plots_background_->bookHistograms();
    mvaOutput_normalization_ = new TFormula(Form("mvaOutput_normalization_%s", mvaName.data()), "1./(1. + [0]*((1./(0.5*TMath::Max(1.e-6, x + 1.))) - 1.)/[1])");
    mvaOutput_normalization_->SetParameter(0, 1.);
    mvaOutput_normalization_->SetParameter(1, 1.);
  }
  ~mvaEntryType() {}
  std::string inputFileName_;
  std::string mvaName_;
  std::string legendEntry_;
  double mvaCut_;
  int tauPtMode_;
  int tauEtaMode_;
  plotEntryType* plots_signal_;
  plotEntryType* plots_background_;
  TFormula* mvaOutput_normalization_;
};

void plotAntiMuonDiscrMVAEfficiency_and_FakeRate()
{
//--- stop ROOT from keeping references to all histograms
  TH1::AddDirectory(false);

//--- suppress the output canvas 
  gROOT->SetBatch(true);

  std::vector<mvaEntryType*> mvaEntries;
  mvaEntries.push_back(new mvaEntryType("/data1/veelken/tmp/antiMuonDiscrMVATraining/antiMuonDiscr_v1_10/trainAntiMuonDiscrMVA_mvaAntiMuonDiscrOpt1.root", "opt1", 0.98, kTauPt, kTauEta));
  mvaEntries.push_back(new mvaEntryType("/data1/veelken/tmp/antiMuonDiscrMVATraining/antiMuonDiscr_v1_10/trainAntiMuonDiscrMVA_mvaAntiMuonDiscrOpt2.root", "opt2", 0.98, kTauPt, kTauAbsEta));
  
  for ( std::vector<mvaEntryType*>::iterator mvaEntry = mvaEntries.begin();
	mvaEntry != mvaEntries.end(); ++mvaEntry ) {
    std::cout << "processing " << (*mvaEntry)->legendEntry_ << std::endl;
    fillPlots((*mvaEntry)->inputFileName_, (*mvaEntry)->plots_signal_, (*mvaEntry)->plots_background_, 
	      (*mvaEntry)->mvaCut_, 
	      (*mvaEntry)->tauPtMode_, (*mvaEntry)->tauEtaMode_, 
	      (*mvaEntry)->mvaOutput_normalization_);
  }
  
  showEfficiency("", 800, 600,
		 mvaEntries[0]->plots_signal_->histogramPt_numerator_, mvaEntries[0]->plots_signal_->histogramPt_denominator_, mvaEntries[0]->legendEntry_,
		 mvaEntries[1]->plots_signal_->histogramPt_numerator_, mvaEntries[1]->plots_signal_->histogramPt_denominator_, mvaEntries[1]->legendEntry_,
		 0, 0, "",
		 0, 0, "",
		 0, 0, "",
		 0, 0, "",
		 "P_{T} / GeV", 1.2, 
		 false, 0.5, 1., "Efficiency", 1.2, 
		 0.61, 0.165, 
		 "plots/plotAntiMuonDiscrMVAEfficiency_vs_Pt.png");
  showEfficiency("", 800, 600,
		 mvaEntries[0]->plots_signal_->histogramEta_numerator_, mvaEntries[0]->plots_signal_->histogramEta_denominator_, mvaEntries[0]->legendEntry_,
		 mvaEntries[1]->plots_signal_->histogramEta_numerator_, mvaEntries[1]->plots_signal_->histogramEta_denominator_, mvaEntries[1]->legendEntry_,
		 0, 0, "",
		 0, 0, "",
		 0, 0, "",
		 0, 0, "",
		 "#eta", 1.2, 
		 false, 0.5, 1., "Efficiency", 1.2, 
		 0.61, 0.165, 
		 "plots/plotAntiMuonDiscrMVAEfficiency_vs_Eta.png");
  showEfficiency("", 800, 600,
		 mvaEntries[0]->plots_signal_->histogramNvtx_numerator_, mvaEntries[0]->plots_signal_->histogramNvtx_denominator_, mvaEntries[0]->legendEntry_,
		 mvaEntries[1]->plots_signal_->histogramNvtx_numerator_, mvaEntries[1]->plots_signal_->histogramNvtx_denominator_, mvaEntries[1]->legendEntry_,
		 0, 0, "",
		 0, 0, "",
		 0, 0, "",
		 0, 0, "",
		 "N_{vtx}", 1.2, 
		 false, 0.5, 1., "Efficiency", 1.2, 
		 0.61, 0.165, 
		 "plots/plotAntiMuonDiscrMVAEfficiency_vs_Nvtx.png");
  
  showEfficiency("", 800, 600,
		 mvaEntries[0]->plots_background_->histogramPt_numerator_, mvaEntries[0]->plots_background_->histogramPt_denominator_, mvaEntries[0]->legendEntry_,
		 mvaEntries[1]->plots_background_->histogramPt_numerator_, mvaEntries[1]->plots_background_->histogramPt_denominator_, mvaEntries[1]->legendEntry_,
		 0, 0, "",
		 0, 0, "",
		 0, 0, "",
		 0, 0, "",
		 "P_{T} / GeV", 1.2, 
		 true, 1.e-3, 1., "Fake-rate", 1.2, 
		 0.61, 0.545,
		 "plots/plotAntiMuonDiscrMVAFakeRate_vs_Pt.png");
  showEfficiency("", 800, 600,
		 mvaEntries[0]->plots_background_->histogramEta_numerator_, mvaEntries[0]->plots_background_->histogramEta_denominator_, mvaEntries[0]->legendEntry_,
		 mvaEntries[1]->plots_background_->histogramEta_numerator_, mvaEntries[1]->plots_background_->histogramEta_denominator_, mvaEntries[1]->legendEntry_,
		 0, 0, "",
		 0, 0, "",
		 0, 0, "",
		 0, 0, "",
		 "#eta", 1.2, 
		 true, 1.e-3, 1., "Fake-rate", 1.2, 
		 0.61, 0.545,
		 "plots/plotAntiMuonDiscrMVAFakeRate_vs_Eta.png");
  showEfficiency("", 800, 600,
		 mvaEntries[0]->plots_background_->histogramNvtx_numerator_, mvaEntries[0]->plots_background_->histogramNvtx_denominator_, mvaEntries[0]->legendEntry_,
		 mvaEntries[1]->plots_background_->histogramNvtx_numerator_, mvaEntries[1]->plots_background_->histogramNvtx_denominator_, mvaEntries[1]->legendEntry_,
		 0, 0, "",
		 0, 0, "",
		 0, 0, "",
		 0, 0, "",
		 "N_{vtx}", 1.2, 
		 true, 1.e-3, 1., "Fake-rate", 1.2, 
		 0.61, 0.545,
		 "plots/plotAntiMuonDiscrMVAFakeRate_vs_Nvtx.png");

  for ( std::vector<mvaEntryType*>::iterator mvaEntry = mvaEntries.begin();
	mvaEntry != mvaEntries.end(); ++mvaEntry ) {

    normalizeHistogram((*mvaEntry)->plots_signal_->histogramPt_numerator_);
    normalizeHistogram((*mvaEntry)->plots_signal_->histogramPt_denominator_);
    normalizeHistogram((*mvaEntry)->plots_signal_->histogramEta_numerator_);
    normalizeHistogram((*mvaEntry)->plots_signal_->histogramEta_denominator_);
    normalizeHistogram((*mvaEntry)->plots_signal_->histogramNvtx_numerator_);
    normalizeHistogram((*mvaEntry)->plots_signal_->histogramNvtx_denominator_);
  
    normalizeHistogram((*mvaEntry)->plots_background_->histogramPt_numerator_);
    normalizeHistogram((*mvaEntry)->plots_background_->histogramPt_denominator_);
    normalizeHistogram((*mvaEntry)->plots_background_->histogramEta_numerator_);
    normalizeHistogram((*mvaEntry)->plots_background_->histogramEta_denominator_);
    normalizeHistogram((*mvaEntry)->plots_background_->histogramNvtx_numerator_);
    normalizeHistogram((*mvaEntry)->plots_background_->histogramNvtx_denominator_);
    
    showDistribution("", 800, 600,
		     (*mvaEntry)->plots_signal_->histogramPt_denominator_, "Signal",
		     (*mvaEntry)->plots_background_->histogramPt_denominator_, "Background",
		     0, "",
		     0, "",
		     "P_{T} / GeV", 1.2, 
		     true, 1.e-3, 1., "a.u.", 1.2, 
		     0.145, 0.745, 
		     TString(Form("plots/plotAntiMuonDiscrMVAEfficiency_and_FakeRate_denominatorPt_%s.png", (*mvaEntry)->legendEntry_.data())).ReplaceAll(" ", "").Data());
    showDistribution("", 800, 600,
		     (*mvaEntry)->plots_signal_->histogramEta_denominator_, "Signal",
		     (*mvaEntry)->plots_background_->histogramEta_denominator_, "Background",
		     0, "",
		     0, "",
		     "#eta", 1.2, 
		     true, 1.e-3, 1., "a.u.", 1.2, 
		     0.145, 0.745, 
		     TString(Form("plots/plotAntiMuonDiscrMVAEfficiency_and_FakeRate_denominatorEta_%s.png", (*mvaEntry)->legendEntry_.data())).ReplaceAll(" ", "").Data());
    showDistribution("", 800, 600,
		     (*mvaEntry)->plots_signal_->histogramNvtx_denominator_, "Signal",
		     (*mvaEntry)->plots_background_->histogramNvtx_denominator_, "Background",
		     0, "",
		     0, "",
		     "N_{vtx}", 1.2, 
		     true, 1.e-3, 1., "a.u.", 1.2, 
		     0.145, 0.745, 
		     TString(Form("plots/plotAntiMuonDiscrMVAEfficiency_and_FakeRate_denominatorNvtx_%s.png", (*mvaEntry)->legendEntry_.data())).ReplaceAll(" ", "").Data());
    
    TGraph* graphEfficiencyEq99_5percent = compMVAcut((*mvaEntry)->plots_signal_->histogramMVAoutput_vs_Pt_, (*mvaEntry)->plots_signal_->histogramPt_, 0.995);
    TGraph* graphEfficiencyEq99_0percent = compMVAcut((*mvaEntry)->plots_signal_->histogramMVAoutput_vs_Pt_, (*mvaEntry)->plots_signal_->histogramPt_, 0.99);
    TGraph* graphEfficiencyEq98_0percent = compMVAcut((*mvaEntry)->plots_signal_->histogramMVAoutput_vs_Pt_, (*mvaEntry)->plots_signal_->histogramPt_, 0.98);

    showGraphs("#tau_{had} Efficiency", 800, 600,
	       graphEfficiencyEq99_5percent, "99.5%",
	       graphEfficiencyEq99_0percent, "99%",
	       graphEfficiencyEq98_0percent, "98%",
	       0, "",
	       0, "",
	       0, "",
	       0., 2500., 10, "P_{T} / GeV", 1.2,
	       0.0, 1.0, "MVA_{cut}", 1.35,
	       0.69, 0.145, 
	       TString(Form("plots/plotAntiMuonDiscrMVAEfficiency_MVAcutVsPtConstEfficiency_%s.png", (*mvaEntry)->legendEntry_.data())).ReplaceAll(" ", "").Data());
    
    std::string outputFileName_MVAoutput_vs_Pt = Form("plots/plotAntiMuonDiscrMVAEfficiency_and_FakeRate_MVAoutput_vs_Pt_%s.root", (*mvaEntry)->mvaName_.data());
    TFile* outputFile_MVAoutput_vs_Pt = new TFile(outputFileName_MVAoutput_vs_Pt.data(), "RECREATE");
    (*mvaEntry)->plots_signal_->histogramMVAoutput_vs_Pt_->Write();
    delete outputFile_MVAoutput_vs_Pt;

    std::string outputFileName_effGraphs = Form("wpDiscriminationAgainstMuonMVA_%s.root", (*mvaEntry)->mvaName_.data());
    graphEfficiencyEq99_5percent->SetName(Form("%seff99_5", (*mvaEntry)->mvaName_.data()));
    graphEfficiencyEq99_0percent->SetName(Form("%seff99_0", (*mvaEntry)->mvaName_.data()));
    graphEfficiencyEq98_0percent->SetName(Form("%seff98_0", (*mvaEntry)->mvaName_.data()));
    TFile* outputFile_effGraphs = new TFile(outputFileName_effGraphs.data(), "RECREATE");
    graphEfficiencyEq99_5percent->Write();
    graphEfficiencyEq99_0percent->Write();
    graphEfficiencyEq98_0percent->Write();
    (*mvaEntry)->mvaOutput_normalization_->Write();			  
    delete outputFile_effGraphs;
    
    TGraph* graphFakeRateEq0001percent = compMVAcut((*mvaEntry)->plots_background_->histogramMVAoutput_vs_Pt_, (*mvaEntry)->plots_background_->histogramPt_, 0.0001);
    TGraph* graphFakeRateEq0002percent = compMVAcut((*mvaEntry)->plots_background_->histogramMVAoutput_vs_Pt_, (*mvaEntry)->plots_background_->histogramPt_, 0.0002);
    TGraph* graphFakeRateEq0005percent = compMVAcut((*mvaEntry)->plots_background_->histogramMVAoutput_vs_Pt_, (*mvaEntry)->plots_background_->histogramPt_, 0.0005);
    TGraph* graphFakeRateEq0010percent = compMVAcut((*mvaEntry)->plots_background_->histogramMVAoutput_vs_Pt_, (*mvaEntry)->plots_background_->histogramPt_, 0.0010);
    TGraph* graphFakeRateEq0020percent = compMVAcut((*mvaEntry)->plots_background_->histogramMVAoutput_vs_Pt_, (*mvaEntry)->plots_background_->histogramPt_, 0.0020);
    TGraph* graphFakeRateEq0050percent = compMVAcut((*mvaEntry)->plots_background_->histogramMVAoutput_vs_Pt_, (*mvaEntry)->plots_background_->histogramPt_, 0.0050);
    
    showGraphs("#tau_{had} Fake-rate", 800, 600,
	       graphFakeRateEq0050percent, "0.5%",
	       graphFakeRateEq0020percent, "0.2%",
	       graphFakeRateEq0010percent, "0.1%",
	       graphFakeRateEq0005percent, "0.05%",
	       graphFakeRateEq0002percent, "0.02%",
	       graphFakeRateEq0001percent, "0.01%",
	       0., 2500., 10, "P_{T} / GeV", 1.2,
	       0.0, 1.0, "MVA_{cut}", 1.35,
	       0.69, 0.145, 
	       TString(Form("plots/plotAntiMuonDiscrMVAEfficiency_MVAcutVsPtConstFakeRate_%s.png", (*mvaEntry)->legendEntry_.data())).ReplaceAll(" ", "").Data());
  }    

  for ( std::vector<mvaEntryType*>::iterator it = mvaEntries.begin();
	it != mvaEntries.end(); ++it ) {
    delete (*it);
  }
}
