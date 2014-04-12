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

void setTDRStyle();

/// Small function to simplify the creation of text in the TPaveText
TString setText(const char * text, const double & num1, const char * divider = "", const double & num2 = 0) {
  std::stringstream numString;
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

class ProjectionFitter
{
 public:
  TGraphErrors * fit2Dprojection(const TString & inputFileName, const TString & histoName,
                                 const unsigned int rebinX, const unsigned int rebinY,
                                 const double & fitXmin, const double & fitXmax,
                                 const TString & fitName, const TString & append,
                                 TFile * outputFile, const unsigned int minEntries = 100);
 protected:
  /// Take the projection and give it a suitable name and title
  TH1 * takeProjectionY( const TH2 * histo, const int index ) {
    // Name
    std::stringstream number;
    number << index;
    TString numberString(number.str());
    TString name = TString(histo->GetName()) + "_" + numberString;
    // Title
    double xBin = histo->GetXaxis()->GetBinCenter(index);
    std::stringstream xBinString;
    xBinString << xBin;
    TString title("Projection of x = ");
    title += xBinString.str();

    TH1 * histoY = histo->ProjectionY(name, index, index);
    histoY->SetName(title);
    return histoY;
  }
};

/// This method fits TH2F slices with a selected fitType function (gaussian, lorentz, ...).
TGraphErrors * ProjectionFitter::fit2Dprojection(const TString & inputFileName, const TString & histoName,
                                                 const unsigned int rebinX, const unsigned int rebinY,
                                                 const double & fitXmin, const double & fitXmax,
                                                 const TString & fitName, const TString & append,
                                                 TFile * outputFile, const unsigned int minEntries)
{
  // Read the TH2 from file
  // std::cout << "inputFileName = " << inputFileName << std::endl;
  TFile *inputFile = new TFile(inputFileName);
  // FindObjectAny finds the object of the given name looking also in the subdirectories.
  // It does not change the current directory if it finds a matching (for that use FindKeyAny).
  TH2 * histo = (TH2*) inputFile->FindObjectAny(histoName);
  if( rebinX > 0 ) histo->RebinX(rebinX);
  if( rebinY > 0 ) histo->RebinY(rebinY);

  std::vector<FitResult> fitResults;

  // Output file
  // TString outputFileName("fitCompare2"+histoName);
  // outputFileName += append;
  // outputFileName += ".root";
  // TFile * outputFile = new TFile(outputFileName,"RECREATE");

  // For each bin in X take the projection on Y and fit.
  // All the fits are saved in a single canvas.
  unsigned int nBins = histo->GetXaxis()->GetNbins();

  std::vector<TH1*> projections;
  std::vector<TF1*> projectionsFits;

  for( unsigned int i=1; i<=nBins; ++i ) {

    // Project on Y
    // ------------
    TH1 * histoY = takeProjectionY( histo, i );

    // Require a minimum number of entries to do the fit
    if( histoY->GetEntries() > minEntries ) {

      // Gaussian fit
      std::stringstream ss;
      ss << i;
      TString fitFunctionName(fitName+"Fit_"+ss.str());
      TF1 * fitFunction = new TF1(fitFunctionName.Data(), fitName.Data(), fitXmin, fitXmax);
      fitFunction->SetLineColor(kRed);
      projectionsFits.push_back(fitFunction);
      // Options: M = "more": improve fit results; Q = "quiet"
      // If not using the N options it will try to save the histogram and corrupt the memory.
      // We save also draw functions and write the later.
      histoY->Fit(fitFunctionName, "MQN", "", fitXmin, fitXmax);

      projections.push_back(histoY);

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

  TCanvas * canvasYcheck = new TCanvas(histoName+"_canvas_check"+append, histoName+" fits check", 1000, 800);
  int sizeCheck = projections.size();
  int x = int(sqrt(sizeCheck));
  int y = x;
  if( x*y < sizeCheck ) y += 1;
  if( x*y < sizeCheck ) x += 1;
  std::cout << "sizeCheck = " << sizeCheck << std::endl;
  std::cout << "x*y = " << x*y << std::endl;
  canvasYcheck->Divide(x, y);

  std::vector<TH1*>::const_iterator it = projections.begin();
  std::vector<TF1*>::const_iterator fit = projectionsFits.begin();
  for( int i=1; it != projections.end(); ++it, ++fit, ++i ) {
    canvasYcheck->cd(i);
    (*it)->Draw();
    (*it)->GetXaxis()->SetRangeUser(fitXmin, fitXmax);
    (*fit)->Draw("same");
    (*fit)->SetLineColor(kRed);
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
  grMean->SetTitle(histoName+"_Mean"+append);
  grMean->SetName(histoName+"_Mean"+append);
  TGraphErrors *grWidth = new TGraphErrors(fitsNumber, xCenter, width, xCenterError, widthError);
  grWidth->SetTitle(histoName+"_Width"+append);
  grWidth->SetName(histoName+"_Width"+append);
  TGraphErrors *grChi2 = new TGraphErrors(fitsNumber, xCenter, chi2, xCenterError, xCenterError);
  grChi2->SetTitle(histoName+"_chi2"+append);
  grChi2->SetName(histoName+"_chi2"+append);

  grMean->SetMarkerColor(4);
  grMean->SetMarkerStyle(20);
  grWidth->SetMarkerColor(4);
  grWidth->SetMarkerStyle(20);
  grChi2->SetMarkerColor(4);
  grChi2->SetMarkerStyle(20);

  outputFile->cd();
  // Draw and save the graphs
  TCanvas * c1 = new TCanvas(histoName+"_Width"+append, histoName+"_Width");
  c1->cd();
  grWidth->Draw("AP");
  c1->Write();
  TCanvas * c2 = new TCanvas(histoName+"_Mean"+append, histoName+"_Mean");
  c2->cd();
  grMean->Draw("AP");
  c2->Write();
  TCanvas * c3 = new TCanvas(histoName+"_Chi2"+append, histoName+"_Chi2");
  c3->cd();
  grChi2->Draw("AP");
  c3->Write();

  // Write the canvas with the fits
  canvasYcheck->Write();

  // outputFile->Close();

  return grMean;
}

/****************************************************************************************/
void macroPlot( TString name, const TString & nameFile1, const TString & nameFile2, const TString & title,
                const TString & resonanceType, const int rebinX, const int rebinY, const TString & outputFileName,
                TFile * externalOutputFile = 0)
{
  gROOT->SetBatch(true);

  //Save the graphs in a file
  TFile *outputFile = externalOutputFile;
  if( outputFile == 0 ) {
    outputFile = new TFile(outputFileName,"RECREATE");
  }

  setTDRStyle();

  ProjectionFitter projectionFitter;

  std::cout << "File 1 = " << nameFile1 << std::endl;
  std::cout << "File 2 = " << nameFile2 << std::endl;

  double y[2];
  if( resonanceType == "JPsi" || resonanceType == "Psi2S" ) { y[0]=2.9; y[1]=3.3; }
  else if( resonanceType.Contains("Upsilon1S") ) { y[0]=9.25; y[1]=9.85; }
  else if( resonanceType.Contains("Upsilon2S") ) { y[0]=9.85; y[1]=10.2; }
  else if( resonanceType.Contains("Upsilon3S") ) { y[0]=10.2; y[1]=10.7; }
  else if( resonanceType == "Z" ) { y[0]=70; y[1]=110; }
  else if( resonanceType == "Resolution" ) { y[0]=-0.3; y[1]=0.3; }

  TGraphErrors *grM_1 = projectionFitter.fit2Dprojection( nameFile1, name, rebinX, rebinY, y[0], y[1], "gaus", "_1", outputFile );
  TGraphErrors *grM_2 = projectionFitter.fit2Dprojection( nameFile2, name, rebinX, rebinY, y[0], y[1], "gaus", "_2", outputFile );

  TCanvas *c = new TCanvas(name,name);
  c->SetGridx();
  c->SetGridy();
    
  grM_1->SetMarkerColor(1);
  grM_2->SetMarkerColor(2);
 
  TString xAxisTitle;

  double x[2];

  if( name.Contains("Eta") ) {
    x[0]=-3; x[1]=3;
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
  if( externalOutputFile == 0 ) {
    outputFile->Close();
  }
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
