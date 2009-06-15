#include "TROOT.h"
#include "TH1.h"
#include "TH2.h"
#include "TF1.h"
#include "TGraphErrors.h"
#include "TLegend.h"
#include "TCanvas.h"
#include "TSystem.h"
#include "TStyle.h"
#include "TFile.h"
#include "TPaveText.h"
#include <vector>
#include <iostream>
#include <sstream>
#include <cmath>
using namespace std;

/**
 * This macro can be used to fit the y projections of 2D histograms. <br>
 * The fit2Dprojection function performs the fits and returns a TGraphErrors
 * with the mean values and their errors. <br>
 *
 * TO-DO:
 * - the values for the fit to the Z are hardcoded. Allow the selection of different intervals.
 * - the only possible fit function is the "gaus", also hardcoded. Allow different functions.
 * - Perform fits on the resulting TGraphErrors with the means.
 */

void setTDRStyle();

/// Small function to simplify the creation of text in the TPaveText
TString setText(const char * text, const double & num1, const char * divider = "", const double & num2 = 0) {
  stringstream numString;
  TString textString(text);
  numString << num1;
  textString += numString.str();
  if( num2 != 0 ) {
    textString += divider;
    numString.str("");
    numString << num2;
    textString += numString.str();
  }
  return textString;
}

/// This function sets up the TPaveText with chi2/ndf and parameters values +- errors
void setTPaveText(const TF1 * fit, TPaveText * paveText) {
  Double_t chi2 = fit->GetChisquare();
  Int_t ndf = fit->GetNDF();
  paveText->AddText(setText("#chi^2 / ndf = ", chi2,  " / ", ndf));

  for( int iPar=0; iPar<fit->GetNpar(); ++iPar ) {
    TString parName(fit->GetParName(iPar));
    Double_t parValue = fit->GetParameter(iPar); // value of Nth parameter
    Double_t parError = fit->GetParError(iPar);  // error on Nth parameter
    paveText->AddText(setText(parName + " = ", parValue, " #pm ", parError));
  }
}

// Structure to hold the fit results
struct FitResult
{
  double top;
  double topError;
  double width;
  double widthError;
  double mean;
  double meanError;
  double xCenter;
  double xCenterError;
  double chi2;
};

/// This function fits TH2F slices with a selected fitType function (gaussian, lorentz, ...).
TGraphErrors * fit2Dprojection(const TString & inputFileName, const TString & histoName,
                               const unsigned int rebinX = 0, const unsigned int rebinY = 0,
                               const double fitXmin = 80, const double fitXmax = 100,
                               const TString & append = "_1", const unsigned int minEntries = 1000) {

  //Read the TH2 from file
  TFile *inputFile = new TFile(inputFileName);
  // FindObjectAny finds the object of the given name looking also in the subdirectories.
  // It does not change the current directory if it finds a matching (for that use FindKeyAny).
  TH2 * histo = (TH2*) inputFile->FindObjectAny(histoName);
  if( rebinX > 0 ) histo->RebinX(rebinX);
  if( rebinY > 0 ) histo->RebinY(rebinY);

  vector<FitResult> fitResults;

  // Output file
  TString outputFileName("fitCompare2"+histoName);
  outputFileName += append;
  outputFileName += ".root";
  TFile * outputFile = new TFile(outputFileName,"RECREATE");

  // For each bin in X take the projection on Y and fit.
  // All the fits are saved in a single canvas.
  unsigned int nBins = histo->GetXaxis()->GetNbins();
  unsigned int canvasYbins = int(sqrt(nBins));
  unsigned int canvasXbins = canvasYbins;
  if( nBins - canvasXbins*canvasYbins > 0 ) {
    canvasXbins += 1;
    canvasYbins += 1;
  }

  TCanvas * canvasY = new TCanvas(histoName+"_canvas", histoName+" fits check", 1000, 800);
  canvasY->Divide(canvasXbins, canvasYbins);
  // canvasY->Draw();
  for( unsigned int i=1; i<=nBins; ++i ) {

    // Project on Y
    // ------------
    // Name
    stringstream number;
    number << i;
    TString numberString(number.str());
    TString name = histoName + "_" + numberString;
    // Title
    double xBin = histo->GetXaxis()->GetBinCenter(i);
    stringstream xBinString;
    xBinString << xBin;
    TString title("Projection of x = ");
    title += xBinString.str();

    TH1 * histoY = histo->ProjectionY(name, i, i);
    histoY->SetName(title);

    // Require a minimum number of entries to do the fit
    if (histoY->GetEntries() > minEntries) {

      // Gaussian fit
      canvasY->cd(i);
      TF1 * fitFunction = new TF1("gaussianFit", "gaus");
      fitFunction->SetLineColor(kRed);
      // Options: M = "more": improve fit results; Q = "quiet"
      histoY->Fit("gaussianFit", "MQ", "", fitXmin, fitXmax);
      // TF1 * fitFunction = histoY->GetFunction("gaus");

      double *par = fitFunction->GetParameters();
      double *err = fitFunction->GetParErrors();

      // Store the fit results
      FitResult fitResult;
      if( par[0] == par[0] ) {
        fitResult.top = par[0];
        fitResult.topError = err[0];
        // sometimes the gaussian has negative width (checked with Rene Brun)
        fitResult.mean = fabs(par[1]);
        fitResult.meanError = err[1];
        fitResult.width = par[2];
        fitResult.widthError = err[2];

        fitResult.xCenter = histo->GetXaxis()->GetBinCenter(i);
        fitResult.xCenterError = 0;

        double chi2 = fitFunction->GetChisquare()/fitFunction->GetNDF();
        if( chi2 == chi2 ) fitResult.chi2 = fitFunction->GetChisquare()/fitFunction->GetNDF();
        else fitResult.chi2 = 100000;
      }
      else {
        // Skip nan
        fitResult.top = 0;
        fitResult.topError = 1;
        // sometimes the gaussian has negative width (checked with Rene Brun)
        fitResult.mean = 0;
        fitResult.meanError = 1;
        fitResult.width = 0;
        fitResult.widthError = 1;

        fitResult.xCenter = histo->GetXaxis()->GetBinCenter(i);
        fitResult.xCenterError = 0;

        fitResult.chi2 = 100000;
      }
      fitResults.push_back(fitResult);
    }
  }

  // Plot the fit results in TGraphs
  const unsigned int fitsNumber = fitResults.size();
  double xCenter[fitsNumber], xCenterError[fitsNumber],
    mean[fitsNumber], meanError[fitsNumber],
    width[fitsNumber], widthError[fitsNumber],
    chi2[fitsNumber];

  double xDisplace = 0;

  for( unsigned int i=0; i<fitsNumber; ++i ) {

    FitResult * fitResult = &(fitResults[i]);

    xCenter[i]      = fitResult->xCenter + xDisplace;
    xCenterError[i] = fitResult->xCenterError;
    mean[i]         = fitResult->mean;
    meanError[i]    = fitResult->meanError;
    width[i]        = fitResult->width;
    width[i]        = fitResult->widthError;
    chi2[i]         = fitResult->chi2;
  }

  TGraphErrors *grMean = new TGraphErrors(fitsNumber, xCenter, mean, xCenterError, meanError);
  grMean->SetTitle(histoName+"_Mean");
  grMean->SetName(histoName+"_Mean");
  TGraphErrors *grWidth = new TGraphErrors(fitsNumber, xCenter, width, xCenterError, widthError);
  grWidth->SetTitle(histoName+"_Width");
  grWidth->SetName(histoName+"_Width");
  TGraphErrors *grChi2 = new TGraphErrors(fitsNumber, xCenter, chi2, xCenterError, xCenterError);
  grChi2->SetTitle(histoName+"_chi2");
  grChi2->SetName(histoName+"_chi2");

  grMean->SetMarkerColor(4);
  grMean->SetMarkerStyle(20);
  grWidth->SetMarkerColor(4);
  grWidth->SetMarkerStyle(20);
  grChi2->SetMarkerColor(4);
  grChi2->SetMarkerStyle(20);

  outputFile->cd();
  // Draw and save the graphs
  TCanvas * c1 = new TCanvas(histoName+"_Width", histoName+"_Width");
  c1->cd();
  grWidth->Draw("AP");
  c1->Write();
  TCanvas * c2 = new TCanvas(histoName+"_Mean", histoName+"_Mean");
  c2->cd();
  grMean->Draw("AP");
  c2->Write();
  TCanvas * c3 = new TCanvas(histoName+"_Chi2", histoName+"_Chi2");
  c3->cd();
  grChi2->Draw("AP");
  c3->Write();

  // Write the canvas with the fits
  canvasY->Write();

  outputFile->Close();

  return grMean;
}

/****************************************************************************************/
void macroPlot( TString name, const TString & nameFile1, const TString & nameFile2, const TString & title,
                const TString & resonanceType, const int rebinX, const int rebinY, const TString & outputFileName) {

  gROOT->SetBatch(true);

  //Save the graphs in a file
  TFile *outputFile = new TFile(outputFileName,"RECREATE");

  setTDRStyle();

  TGraphErrors *grM_1 = fit2Dprojection( nameFile1, name, rebinX, rebinY, 70, 110, "_1" );
  TGraphErrors *grM_2 = fit2Dprojection( nameFile2, name, rebinX, rebinY, 70, 110, "_2" );

  TCanvas *c = new TCanvas(name+"_Z",name+"_Z");
  c->SetGridx();
  c->SetGridy();
    
  grM_1->SetMarkerColor(1);
  grM_2->SetMarkerColor(2);
 
  TString xAxisTitle;

  double x[2],y[2];

  if( name.Contains("Eta") ) {
    x[0]=-3; x[1]=3;       //<------useful for reso VS eta
    xAxisTitle = "#eta";
  }
  else if( name.Contains("PhiPlus") || name.Contains("PhiMinus") ) {
    x[0] = -3.2; x[1] = 3.2;
    xAxisTitle = "#phi(rad)";
  }
  else {
    x[0] = 0.; x[1] = 200;
    xAxisTitle = "pt(GeV)";
  }
  if( resonanceType == "JPsi" || resonanceType == "Psi2S" ) { y[0]=0.; y[1]=6.; }
  else if( resonanceType.Contains("Upsilon") ) { y[0]=8.; y[1]=12.; }
  else if( resonanceType == "Z" ) { y[0]=80; y[1]=100; }

  // This is used to have a canvas containing both histogram points
  TGraph *gr = new TGraph(2,x,y);
  gr->SetMarkerStyle(8);
  gr->SetMarkerColor(108);
  gr->GetYaxis()->SetTitle("Mass(GeV)   ");
  gr->GetYaxis()->SetTitleOffset(1);
  gr->GetXaxis()->SetTitle(xAxisTitle);
  gr->SetTitle(title);

  // Text for the fits
  TPaveText * paveText1 = new TPaveText(0.20,0.15,0.49,0.35,"NDC");
  paveText1->SetFillColor(0);
  paveText1->SetTextColor(1);
  paveText1->SetTextSize(0.02);
  paveText1->SetBorderSize(1);
  TPaveText * paveText2 = new TPaveText(0.59,0.15,0.88,0.35,"NDC");
  paveText2->SetFillColor(0);
  paveText2->SetTextColor(2);
  paveText2->SetTextSize(0.02);
  paveText2->SetBorderSize(1);

  /*****************************************/ 
  if( name.Contains("Pt") ) {
    cout << "Fitting pt" << endl;
    // TF1 * fit1 = new TF1("fit1", linear, 0., 150., 2);
    TF1 * fit1 = new TF1("fit1", "[0]", 0., 150.);
    fit1->SetParameters(0., 1.);
    fit1->SetParNames("scale","pt coefficient");
    fit1->SetLineWidth(2);
    fit1->SetLineColor(1);
    if( grM_1->GetN() > 0 ) {
      if( name.Contains("Z") ) grM_1->Fit("fit1","","",0.,150.);
      else grM_1->Fit("fit1","","",0.,27.);
    }
    setTPaveText(fit1, paveText1);

    // TF1 * fit2 = new TF1("fit2", linear, 0., 150., 2);
    TF1 * fit2 = new TF1("fit2", "[0]", 0., 150.);
    fit2->SetParameters(0., 1.);
    fit2->SetParNames("scale","pt coefficient");
    fit2->SetLineWidth(2);
    fit2->SetLineColor(2);
    if( grM_2->GetN() > 0 ) {
      if( name.Contains("Z") ) grM_2->Fit("fit2","","",0.,150.);
      grM_2->Fit("fit2","","",0.,27.);
    }
    setTPaveText(fit2, paveText2);
  }

  c->cd();
  gr->Draw("AP");
  grM_1->Draw("P");
  grM_2->Draw("P");

  paveText1->Draw("same");
  paveText2->Draw("same");

  TLegend *leg = new TLegend(0.65,0.85,1,1);
  leg->SetFillColor(0);
  leg->AddEntry(grM_1,"before calibration","P");
  leg->AddEntry(grM_2,"after calibration","P");
  leg->Draw("same");

  outputFile->cd();
  c->Write();
  outputFile->Close();
}

/****************************************************************************************/

void setTDRStyle() {
  TStyle *tdrStyle = new TStyle("tdrStyle","Style for P-TDR");

  // For the canvas:
  tdrStyle->SetCanvasBorderMode(0);
  tdrStyle->SetCanvasColor(kWhite);
  tdrStyle->SetCanvasDefH(600); //Height of canvas
  tdrStyle->SetCanvasDefW(600); //Width of canvas
  tdrStyle->SetCanvasDefX(0);   //POsition on screen
  tdrStyle->SetCanvasDefY(0);

  // For the Pad:
  tdrStyle->SetPadBorderMode(0);
  tdrStyle->SetPadColor(kWhite);
  tdrStyle->SetPadGridX(false);
  tdrStyle->SetPadGridY(false);
  tdrStyle->SetGridColor(0);
  tdrStyle->SetGridStyle(3);
  tdrStyle->SetGridWidth(1);

  // For the frame:
  tdrStyle->SetFrameBorderMode(0);
  tdrStyle->SetFrameBorderSize(1);
  tdrStyle->SetFrameFillColor(0);
  tdrStyle->SetFrameFillStyle(0);
  tdrStyle->SetFrameLineColor(1);
  tdrStyle->SetFrameLineStyle(1);
  tdrStyle->SetFrameLineWidth(1);

  // For the histo:
  tdrStyle->SetHistLineColor(1);
  tdrStyle->SetHistLineStyle(0);
  tdrStyle->SetHistLineWidth(1);

  tdrStyle->SetEndErrorSize(2);
  tdrStyle->SetErrorX(0.);
  
  tdrStyle->SetMarkerStyle(20);

  //For the fit/function:
  tdrStyle->SetOptFit(1);
  tdrStyle->SetFitFormat("5.4g");
  tdrStyle->SetFuncColor(2);
  tdrStyle->SetFuncStyle(1);
  tdrStyle->SetFuncWidth(1);

  //For the date:
  tdrStyle->SetOptDate(0);

  // For the statistics box:
  tdrStyle->SetOptFile(0);
  tdrStyle->SetOptStat(0); // To display the mean and RMS:   SetOptStat("mr");
  tdrStyle->SetStatColor(kWhite);
  tdrStyle->SetStatFont(42);
  tdrStyle->SetStatFontSize(0.025);
  tdrStyle->SetStatTextColor(1);
  tdrStyle->SetStatFormat("6.4g");
  tdrStyle->SetStatBorderSize(1);
  tdrStyle->SetStatH(0.1);
  tdrStyle->SetStatW(0.15);

  // Margins:
  tdrStyle->SetPadTopMargin(0.05);
  tdrStyle->SetPadBottomMargin(0.13);
  tdrStyle->SetPadLeftMargin(0.13);
  tdrStyle->SetPadRightMargin(0.05);

  // For the Global title:
  tdrStyle->SetOptTitle(0);

  // For the axis titles:
  tdrStyle->SetTitleColor(1, "XYZ");
  tdrStyle->SetTitleFont(42, "XYZ");
  tdrStyle->SetTitleSize(0.06, "XYZ");
  tdrStyle->SetTitleXOffset(0.9);
  tdrStyle->SetTitleYOffset(1.05);

  // For the axis labels:
  tdrStyle->SetLabelColor(1, "XYZ");
  tdrStyle->SetLabelFont(42, "XYZ");
  tdrStyle->SetLabelOffset(0.007, "XYZ");
  tdrStyle->SetLabelSize(0.05, "XYZ");

  // For the axis:
  tdrStyle->SetAxisColor(1, "XYZ");
  tdrStyle->SetStripDecimals(kTRUE);
  tdrStyle->SetTickLength(0.03, "XYZ");
  tdrStyle->SetNdivisions(510, "XYZ");
  tdrStyle->SetPadTickX(1);  // To get tick marks on the opposite side of the frame
  tdrStyle->SetPadTickY(1);

  // Change for log plots:
  tdrStyle->SetOptLogx(0);
  tdrStyle->SetOptLogy(0);
  tdrStyle->SetOptLogz(0);

  //tdrStyle->SetOptFit(00000);
  tdrStyle->cd();
}
