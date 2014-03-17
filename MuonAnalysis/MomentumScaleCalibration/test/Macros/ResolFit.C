#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <list>

#include "TROOT.h"
#include "TH1D.h"
#include "TF1.h"
#include "TCanvas.h"
#include "TPaveText.h"
#include "TFile.h"

/**
 * This is the pt vs eta resolution by points. It uses fabs(eta) assuming symmetry.
 */
Double_t etaByPoints(Double_t * inEta, Double_t * par) {
  Double_t sigmaPtVsEta = 0.;
  Double_t eta = fabs(inEta[0]);
  if( 0. <= eta && eta <= 0.2 ) sigmaPtVsEta = 0.0120913;
  else if( 0.2 < eta && eta <= 0.4 ) sigmaPtVsEta = 0.0122204;
  else if( 0.4 < eta && eta <= 0.6 ) sigmaPtVsEta = 0.0136937;
  else if( 0.6 < eta && eta <= 0.8 ) sigmaPtVsEta = 0.0142069;
  else if( 0.8 < eta && eta <= 1.0 ) sigmaPtVsEta = 0.0177526;
  else if( 1.0 < eta && eta <= 1.2 ) sigmaPtVsEta = 0.0243587;
  else if( 1.2 < eta && eta <= 1.4 ) sigmaPtVsEta = 0.019994;
  else if( 1.4 < eta && eta <= 1.6 ) sigmaPtVsEta = 0.0185132;
  else if( 1.6 < eta && eta <= 1.8 ) sigmaPtVsEta = 0.0177141;
  else if( 1.8 < eta && eta <= 2.0 ) sigmaPtVsEta = 0.0211577;
  else if( 2.0 < eta && eta <= 2.2 ) sigmaPtVsEta = 0.0255051;
  else if( 2.2 < eta && eta <= 2.4 ) sigmaPtVsEta = 0.0338104;
  // ATTENTION: This point has a big error and it is very displaced from the rest of the distribution.
  else if( 2.4 < eta && eta <= 2.6 ) sigmaPtVsEta = 0.31;
  return ( par[0]*sigmaPtVsEta );
}

void myFunc()
{
  TF1 *f1 = new TF1("myFunc",etaByPoints,-2.5,0,1);
  f1->SetParameter(0, 1);
  f1->SetParNames("constant");
  // f1->Draw();
}

/**
 * This function draws a histogram and a function on a canvas and adds a text box with the results of the fit.
 */
void draw(TH1D * h, TF1 * f) {
  Width_t lineWidth = Width_t(0.4);
  Color_t lineColor = kRed;
  f->SetLineColor(lineColor);
  f->SetLineWidth(lineWidth);
  TCanvas c(h->GetName(),h->GetTitle(),1000,800);
  c.cd();
  h->Draw();
  f->Draw("same");

  // Text box with fit results
  TPaveText * fitLabel = new TPaveText(0.78,0.71,0.98,0.91,"NDC");
  fitLabel->SetBorderSize(1);
  fitLabel->SetTextAlign(12);
  fitLabel->SetTextSize(0.02);
  fitLabel->SetFillColor(0);
  fitLabel->AddText("Function: "+f->GetExpFormula());
  for( int i=0; i<f->GetNpar(); ++i ) {
    char name[50];
    std::cout << "par["<<i<<"] = " << f->GetParameter(i) << std::endl;
    sprintf(name, "par[%i] = %4.2g #pm %4.2g",i, f->GetParameter(i), f->GetParError(i));
    fitLabel->AddText(name);
  }
  fitLabel->Draw("same");

  c.Write();
  h->Write();
}

/**
 * This function reads parameters from the FitParameters.txt file and returns them in a vector.
 */
pair<list<double>, list<double> > readParameters(int fitFile) {
  list<double> parameters;
  list<double> parameterErrors;

  std::ifstream a("FitParameters.txt");
  std::string line;
  bool indexFound = false;
  std::string iteration("Iteration ");
  while (a) {
    getline(a,line);
    unsigned int lineInt = line.find("value");

    // Take only the values from the matching iteration
    if( line.find(iteration) != std::string::npos ) {
      std::stringstream iterationNum;
      iterationNum << fitFile;
      if( line.find(iteration+iterationNum.str()) != std::string::npos ) {
        indexFound = true;
        std::cout << "In: " << line << std::endl;
        std::cout << "Found Index = " << iteration+iterationNum.str() << std::endl;
      }
      else indexFound = false;
    }

    if ( (lineInt != std::string::npos) && indexFound ) {
      int subStr1 = line.find("value");
      int subStr2 = line.find("+-");
      std::stringstream paramStr;
      double param = 0;
      paramStr << line.substr(subStr1+5,subStr2);
      paramStr >> param;
      parameters.push_back(param);
      // std::cout << "paramStr = " << line.substr(subStr1+5,subStr2) << std::endl;
      std::stringstream parErrorStr;
      double parError = 0;
      parErrorStr << line.substr(subStr2+1);
      parErrorStr >> parError;
      parameterErrors.push_back(parError);

      // std::cout << "param = " << param << std::endl;
      // std::cout << "parError = " << parError << std::endl;
    }
  }

  //     std::cout << "Reading function from file" << std::endl;
  //     TString param = "aaa a a a  a value = 193.4+-12";
  //     int id = param.Index("value");
  //     int length = param.Length();
  //     std::cout << "param(id,-1)" << param(id, length) << std::endl;
  return make_pair(parameters, parameterErrors);
}

/**
 * Set parameters from the list to the function. It empties the list while using it.
 */
void setParameters(TF1 * f, pair<list<double>, list<double> > & parameters) {
  int parNum = f->GetNpar();
  for( int iPar=0; iPar<parNum; ++iPar ) {
    // Read the first element and remove it from the list
    f->SetParameter(iPar, parameters.first.front());
    f->SetParError(iPar, parameters.second.front());
    parameters.first.pop_front();
    parameters.second.pop_front();
  }
}

/**
 * This macro fits and draws the histograms in the file written by ResolDraw.cc. 
 * If the false is passed as argument, it searches for the file FitParameters, reads
 * the parameters of the functions to draw over the histograms instead of fitting.
 * ATTENTION: the functions used in this case must be the same of those which parameters
 * have been written in FitParameters.txt.
 */
int ResolFit( int fitFile = -1 ) {

  TString mainPtName("PtResolution");
  TString mainCotgThetaName("CotgThetaResolution");
  TString mainPhiName("PhiResolution");

  // Read the values from the FitParameters.txt file if required
  pair<list<double>, list<double> > parameters;
  if( fitFile != -1 ) {
    parameters = readParameters( fitFile );
  }

  TFile inputFile("redrawed.root","READ");
  TFile outputFile("fitted.root","RECREATE");

  outputFile.cd();

  // Pt resolution
  // -------------
  // VS pt
  TDirectory * tempDir = (TDirectory*) inputFile.Get(mainPtName+"GenVSMu");
  TH1D * h = (TH1D*) tempDir->Get(mainPtName+"GenVSMu_ResoVSPt_resol");
  TF1 *f = new TF1("f","pol1",0,100);

  if( fitFile == -1 ) {
    std::cout << "Fitting Pt resolution vs Pt" << std::endl;
    h->Fit("f","R0");
  }
  else {
    setParameters(f, parameters);
    // Put back the constant so that it can be used also by the following function
    parameters.first.push_front(f->GetParameter(0));
    parameters.second.push_front(f->GetParError(0));
  }

  h->SetMinimum(0);
  h->SetMaximum(0.045);
  h->GetXaxis()->SetTitle("pt(GeV)");
  h->GetYaxis()->SetTitleOffset(1.4);
  h->GetYaxis()->SetTitle("#sigma pt");
  draw(h,f);

  std::cout << "sigmapt vs eta" << std::endl;
  // VS eta
  tempDir = (TDirectory*) inputFile.Get(mainPtName+"GenVSMu");
  h = (TH1D*) tempDir->Get(mainPtName+"GenVSMu_ResoVSEta_resol");

  // f = new TF1("f","pol2",-2.5,2.5);

  // This call is needed in order to use myFunc in the fit.
  myFunc();
  f = (TF1*)gROOT->GetFunction("myFunc");
  f->SetParameter(0,1.);

  if( fitFile == -1 ) {
    std::cout << "Fitting Pt resolution vs Eta" << std::endl;
    // h->Fit("f","R0");
    h->Fit("myFunc","R0");
  }
  else {
    setParameters(f, parameters);
  }

  h->SetMinimum(0);
  h->SetMaximum(0.045);
  h->GetXaxis()->SetTitle("#eta");
  h->GetYaxis()->SetTitleOffset(1.4);
  h->GetYaxis()->SetTitle("#sigma pt");
  draw(h,f);

  // CotgTheta resolution
  // --------------------
  // VS pt
  tempDir = (TDirectory*) inputFile.Get(mainCotgThetaName+"GenVSMu");
  h = (TH1D*) tempDir->Get(mainCotgThetaName+"GenVSMu_ResoVSPt_resol");

  f = new TF1("f","[0]+[1]/x",0,100);
  if( fitFile == -1 ) {
    h->Fit("f","R0");
    std::cout << "Fitting CotgTheta resolution vs Pt" << std::endl;
  }
  else {
    setParameters(f, parameters);
    parameters.first.push_front(f->GetParameter(0));
    parameters.second.push_front(f->GetParError(0));
  }

  h->SetMinimum(0);
  h->SetMaximum(0.0014);
  h->GetXaxis()->SetTitle("pt(GeV)");
  h->GetYaxis()->SetTitleOffset(1.4);
  h->GetYaxis()->SetTitle("#sigma cotg#theta");
  draw(h,f);
  // VS eta
  tempDir = (TDirectory*) inputFile.Get(mainCotgThetaName+"GenVSMu");
  h = (TH1D*) tempDir->Get(mainCotgThetaName+"GenVSMu_ResoVSEta_resol");

  f = new TF1("f","pol2",-2.5,2.5);
  if( fitFile == -1 ) {
    std::cout << "Fitting CotgTheta resolution vs Eta" << std::endl;
    h->Fit("f","R0");
  }
  else {
    setParameters(f, parameters);
  }

  h->SetMinimum(0);
  h->SetMaximum(0.0035);
  h->GetXaxis()->SetTitle("eta");
  h->GetYaxis()->SetTitleOffset(1.4);
  h->GetYaxis()->SetTitle("#sigma cotg#theta");
  draw(h,f);

  // Phi resolution
  // --------------
  // VS pt
  tempDir = (TDirectory*) inputFile.Get(mainPhiName+"GenVSMu");
  h = (TH1D*) tempDir->Get(mainPhiName+"GenVSMu_ResoVSPt_resol");

  f = new TF1("f","[0]+[1]/x",0,100);
  if( fitFile == -1 ) {
    std::cout << "Fitting Phi resolution vs Pt" << std::endl;
    h->Fit("f","R0");
  }
  else {
    setParameters(f, parameters);
    parameters.first.push_front(f->GetParameter(0));
    parameters.second.push_front(f->GetParError(0));
  }

  h->SetMinimum(0);
  h->SetMaximum(0.001);
  h->GetXaxis()->SetTitle("pt(GeV)");
  h->GetYaxis()->SetTitleOffset(1.4);
  h->GetYaxis()->SetTitle("#sigma #phi");
  draw(h,f);
  // VS eta
  tempDir = (TDirectory*) inputFile.Get(mainPhiName+"GenVSMu");
  h = (TH1D*) tempDir->Get(mainPhiName+"GenVSMu_ResoVSEta_resol");

  f = new TF1("f","pol2",-2.4,2.4);
  if( fitFile == -1 ) {
    std::cout << "Fitting Phi resolution vs Eta" << std::endl;
    h->Fit("f","R0");
  }
  else {
    setParameters(f, parameters);
  }

  h->SetMinimum(0);
  h->SetMaximum(0.005);
  h->GetXaxis()->SetTitle("eta");
  h->GetYaxis()->SetTitleOffset(1.4);
  h->GetYaxis()->SetTitle("#sigma #phi");
  draw(h,f);
  return 0;
}
