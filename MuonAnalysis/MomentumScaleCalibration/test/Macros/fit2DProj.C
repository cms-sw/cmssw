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
#include <iomanip>


/**
 * Small function to simplify the creation of text in the TPaveText. <br>
 * It automatically writes the corret number of figures.
 */
TString setText(const char * text, const double & num1, const char * divider = "", const double & num2 = 0)
{
  // std::cout << "text = " << text << ", num1 = " << num1 << ", divider = " << divider << ", num2 = " << num2 << std::endl;

  // Counter gives the precision
  int precision = 1;
  int k=1;
  while( int(num2*k) == 0 ) {
    // std::cout << "int(num2*"<<k<<")/int("<<k<<") = " << int(num2*k)/int(k) << std::endl;
    k*=10;
    ++precision;
  }

  std::stringstream numString;
  TString textString(text);
  numString << std::setprecision(precision) << std::fixed << num1;
  textString += numString.str();
  if( num2 != 0 ) {
    textString += divider;
    numString.str("");
    if( std::string(text).find("ndf") != std::string::npos ) precision = 0;
    numString << std::setprecision(precision) << std::fixed << num2;
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

TGraphErrors* fit2DProj(TString name, TString path, int minEntries, int rebinX, int rebinY, int fitType,
                        TFile * outputFile, const TString & resonanceType = "Upsilon", const double & xDisplace = 0., const TString & append = "");
void macroPlot( TString name, TString nameGen, const TString & nameFile1 = "0_MuScleFit.root", const TString & nameFile2 = "4_MuScleFit.root",
                const TString & title = "", const TString & resonanceType = "Upsilon", const int rebinX = 0, const int rebinY = 0, const int fitType = 1, const TString & outputFileName = "filegraph.root", const bool genMass = false );

Double_t gaussian(Double_t *x, Double_t *par);
Double_t lorentzian(Double_t *x, Double_t *par);
Double_t lorentzianPlusLinear(Double_t *x, Double_t *par);

Double_t sinusoidal(Double_t *x, Double_t *par);
Double_t parabolic(Double_t *x, Double_t *par);
Double_t parabolic2(Double_t *x, Double_t *par);
Double_t onlyParabolic(Double_t *x, Double_t *par);
Double_t linear(Double_t *x, Double_t *par);
Double_t overX(Double_t *x, Double_t *par);

TF1* gaussianFit(TH1* histoY, const TString & resonanceType = "Upsilon");
TF1* lorentzianFit(TH1* histoY, const TString & resonanceType = "Upsilon");
TF1* linLorentzianFit(TH1* histoY);

void setTDRStyle();
// Gaussian function    
Double_t gaussian(Double_t *x, Double_t *par) {
  return par[0]*exp(-0.5*((x[0]-par[2])/par[1])*((x[0]-par[2])/par[1]));
}
// Lorentzian function
Double_t lorentzian(Double_t *x, Double_t *par) {
  return (0.5*par[0]*par[1]/3.14) /
    TMath::Max( 1.e-10,(x[0]-par[2])*(x[0]-par[2]) + .25*par[1]*par[1]);
}
// Linear + Lorentzian function
Double_t lorentzianPlusLinear(Double_t *x, Double_t *par) {
  return  ((0.5*par[0]*par[1]/3.14)/
    TMath::Max(1.e-10,((x[0]-par[2])*(x[0]-par[2])+.25*par[1]*par[1])))+par[3]+par[4]*x[0];
}

/**
 * This function fits TH2F slices with a selected fitType function (gaussian, lorentz, ...).
 */
TGraphErrors* fit2DProj(TString name, TString path, int minEntries, int rebinX, int rebinY, int fitType,
                        TFile * outputFile, const TString & resonanceType, const double & xDisplace, const TString & append) {

  //Read the TH2 from file
  TFile *inputFile = new TFile(path);
  TH2 * histo = (TH2*) inputFile->Get(name);
  if( rebinX > 0 ) histo->RebinX(rebinX);
  if( rebinY > 0 ) histo->RebinY(rebinY);

  //Declare some variables
  TH1 * histoY;
  TString nameY;
  std::vector<double> Ftop;
  std::vector<double> Fwidth;
  std::vector<double> Fmass;
  std::vector<double> Etop;
  std::vector<double> Ewidth;
  std::vector<double> Emass;
  std::vector<double> Fchi2;
  std::vector<double> Xcenter;
  std::vector<double> Ex;

  TString fileOutName("fitCompare2"+name);
  fileOutName += append;
  fileOutName += ".root";
  TFile *fileOut=new TFile(fileOutName,"RECREATE");

  for (int i=1; i<=(histo->GetXaxis()->GetNbins());i++) {

    //Project on Y (set name and title)
    std::stringstream number;
    number << i;
    TString numberString(number.str());
    nameY = name + "_" + numberString;
    // std::cout << "nameY  " << nameY << std::endl;

    histoY = histo->ProjectionY(nameY, i, i);

    double xBin = histo->GetXaxis()->GetBinCenter(i);
    std::stringstream xBinString;
    xBinString << xBin;
    TString title("Projection of x = ");
    title += xBinString.str();

    histoY->SetName(title);

    if (histoY->GetEntries() > minEntries) {

      //Make the dirty work!
      TF1 *fit;
      std::cout << "fitType = " << fitType << std::endl;
      if(fitType == 1) fit = gaussianFit(histoY, resonanceType);
      else if(fitType == 2) fit = lorentzianFit(histoY, resonanceType);
      else if(fitType == 3) fit = linLorentzianFit(histoY);
      else {
	std::cout<<"Wrong fit type: 1=gaussian, 2=lorentzian, 3=lorentzian+linear."<<std::endl;
	abort();
      }

      double *par = fit->GetParameters();
      double *err = fit->GetParErrors();

      // Check the histogram alone
      TCanvas *canvas = new TCanvas(nameY+"alone", nameY+" alone");
      histoY->Draw();
      canvas->Write();

      // Only for check
      TCanvas *c = new TCanvas(nameY, nameY);

      histoY->Draw();
      fit->Draw("same");
      fileOut->cd();
      c->Write();

      if( par[0] == par[0] ) {
        //Store the fit results
        Ftop.push_back(par[0]);
        Fwidth.push_back(fabs(par[1]));//sometimes the gaussian has negative width (checked with Rene Brun)
        Fmass.push_back(par[2]);
        Etop.push_back(err[0]);
        Ewidth.push_back(err[1]);
        Emass.push_back(err[2]);

        Fchi2.push_back(fit->GetChisquare()/fit->GetNDF());

        double xx= histo->GetXaxis()->GetBinCenter(i);
        Xcenter.push_back(xx);
        double ex = 0;
        Ex.push_back(ex); 
      }
      else {
        // Skip nan
        std::cout << "Skipping nan" << std::endl;
        Ftop.push_back(0);
        Fwidth.push_back(0);
        Fmass.push_back(0);
        Etop.push_back(1);
        Ewidth.push_back(1);
        Emass.push_back(1);

        Fchi2.push_back(100000);

        Xcenter.push_back(0);
        Ex.push_back(1); 
      }
    }
  }

  fileOut->Close();

  //Plots the fit results in  TGraphs
  const int nn= Ftop.size();                   
  double x[nn],ym[nn],e[nn],eym[nn];
  double yw[nn],eyw[nn],yc[nn];

  // std::cout << "number of bins = " << nn << std::endl;
  // std::cout << "Values:" << std::endl;

  for (int j=0;j<nn;j++){
    // std::cout << "xCenter["<<j<<"] = " << Xcenter[j] << std::endl;
    x[j]=Xcenter[j]+xDisplace;
    // std::cout << "Fmass["<<j<<"] = " << Fmass[j] << std::endl;
    ym[j]=Fmass[j];
    // std::cout << "Emass["<<j<<"] = " << Emass[j] << std::endl;
    eym[j]=Emass[j];
    // std::cout << "Fwidth["<<j<<"] = " << Fwidth[j] << std::endl;
    yw[j]=Fwidth[j];
    // std::cout << "Ewidth["<<j<<"] = " << Ewidth[j] << std::endl;
    eyw[j]=Ewidth[j];
    // std::cout << "Fchi2["<<j<<"] = " << Fchi2[j] << std::endl;
    yc[j]=Fchi2[j];
    e[j]=0;
  }

  TGraphErrors *grM = new TGraphErrors(nn,x,ym,e,eym);
  grM->SetTitle(name+"_M");
  grM->SetName(name+"_M");
  TGraphErrors *grW = new TGraphErrors(nn,x,yw,e,eyw);
  grW->SetTitle(name+"_W");
  grW->SetName(name+"_W");
  TGraphErrors *grC = new TGraphErrors(nn,x,yc,e,e);
  grC->SetTitle(name+"_chi2");
  grC->SetName(name+"_chi2");

  grM->SetMarkerColor(4);
  grM->SetMarkerStyle(20);
  grW->SetMarkerColor(4);
  grW->SetMarkerStyle(20);
  grC->SetMarkerColor(4);
  grC->SetMarkerStyle(20);

  //Draw and save the graphs
  outputFile->cd();
  TCanvas * c1 = new TCanvas(name+"_W",name+"_W");
  c1->cd();
  grW->Draw("AP");
  c1->Write();
  TCanvas * c2 = new TCanvas(name+"_M",name+"_M");
  c2->cd();
  grM->Draw("AP");
  c2->Write();
  TCanvas * c3 = new TCanvas(name+"_C",name+"_C");
  c3->cd();
  grC->Draw("AP");
  c3->Write();

  return grM;
}

TF1* gaussianFit(TH1* histoY, const TString & resonanceType){

  TString name = histoY->GetName() + TString("Fit");

  // Fit slices projected along Y from bins in X
  // -------------------------------------------
  TF1 *fit = 0;
  // Set parameters according to the selected resonance
  if( resonanceType == "JPsi" ) {
    fit = new TF1(name,gaussian,2,4,3);
    fit->SetParLimits(2, 3.09, 3.15);
  }
  if( resonanceType.Contains("Upsilon") ) {
    fit = new TF1(name,gaussian,8.5,10.5,3);
    // fit = new TF1(name,gaussian,9,11,3);
    fit->SetParLimits(2, 9.2, 9.6);
    fit->SetParLimits(1, 0.09, 0.1);
  }
  if( resonanceType == "Z" ) {
    fit = new TF1(name,gaussian,80,100,3);
    fit->SetParLimits(2, 80, 100);
    fit->SetParLimits(1, 0.01, 1);
  }
  if( resonanceType == "reso" ) {
    fit = new TF1(name,gaussian,-0.05,0.05,3);
    fit->SetParLimits(2, -0.5, 0.5);
    fit->SetParLimits(1, 0, 0.5);
  }
  fit->SetParameters(histoY->GetMaximum(),histoY->GetRMS(),histoY->GetMean());
  // fit->SetParLimits(1, 0.01, 1);
  //   fit->SetParLimits(1, 40, 60);
  fit->SetParNames("norm","width","mean");
  fit->SetLineWidth(2);

  if( histoY->GetNbinsX() > 1000 ) histoY->Rebin(10);
  histoY->Fit(name,"R0");

  return fit;
}

TF1* lorentzianFit(TH1* histoY, const TString & resonanceType){
  TString name = histoY->GetName() + TString("Fit");

  // Fit slices projected along Y from bins in X 
  TF1 *fit = 0;
  if( resonanceType == "JPsi" || resonanceType == "Psi2S" ) {
    fit = new TF1(name, lorentzian, 3.09, 3.15, 3);
  }
  if( resonanceType.Contains("Upsilon") ) {
    fit = new TF1(name, lorentzian, 9, 10, 3);
  }
  if( resonanceType == "Z" ) {
    fit = new TF1(name, lorentzian, 80, 105, 3);
  }
  fit->SetParameters(histoY->GetMaximum(),histoY->GetRMS(),histoY->GetMean());
  fit->SetParNames("norm","width","mean");
  fit->SetLineWidth(2);
  histoY->Fit( name,"R0" );
  return fit;
}

TF1* linLorentzianFit(TH1* histoY){
  TString name = histoY->GetName() + TString("Fit");

  // Fit slices projected along Y from bins in X 
  TF1 *fit = new TF1(name,lorentzianPlusLinear,70,110,5);
  fit->SetParameters(histoY->GetMaximum(),histoY->GetRMS(),histoY->GetMean(),10,-0.1);
  fit->SetParNames("norm","width","mean","offset","slope");
  fit->SetParLimits(1,-10,10);
  //fit->SetParLimits(2,85,95);
  //fit->SetParLimits(2,90,93);
  fit->SetParLimits(2,2,4);
  fit->SetParLimits(3,0,100);
  fit->SetParLimits(4,-1,0);
  fit->SetLineWidth(2);
  //histoY -> Fit(name,"0","",85,97);
  histoY -> Fit(name,"0","",2,4);
  return fit;
}

/****************************************************************************************/
void macroPlot( TString name, TString nameGen, const TString & nameFile1, const TString & nameFile2, const TString & title,
                const TString & resonanceType, const int rebinX, const int rebinY, const int fitType,
                const TString & outputFileName, const bool genMass) {

  gROOT->SetBatch(true);

  //Save the graphs in a file
  TFile *outputFile = new TFile(outputFileName,"RECREATE");

  setTDRStyle();

  TGraphErrors *grM_1 = fit2DProj(name, nameFile1, 100, rebinX, rebinY, fitType, outputFile, resonanceType, 0, "_1");
  TGraphErrors *grM_2 = fit2DProj(name, nameFile2, 100, rebinX, rebinY, fitType, outputFile, resonanceType, 0, "_2");
  TGraphErrors *grM_Gen = fit2DProj(nameGen, nameFile2, 100, rebinX, rebinY, fitType, outputFile, resonanceType, 0, "");

  TCanvas *c = new TCanvas(name+"_Z",name+"_Z");
  c->SetGridx();
  c->SetGridy();
    
  grM_1->SetMarkerColor(1);
  grM_2->SetMarkerColor(2);
  if(genMass)
    grM_Gen->SetMarkerColor(4);
 
  TString xAxisTitle;

  double x[2],y[2];

  if( name.Contains("Eta") ) {
    x[0]=-3; x[1]=3;       //<------useful for reso VS eta
    xAxisTitle = "muon #eta";
  }
  else if( name.Contains("PhiPlus") || name.Contains("PhiMinus") ) {
    x[0] = -3.2; x[1] = 3.2;
    xAxisTitle = "muon #phi(rad)";
  }
  else {
    x[0] = 0.; x[1] = 200;
    xAxisTitle = "muon pt(GeV)";
  }
  if( resonanceType == "JPsi" || resonanceType == "Psi2S" ) { y[0]=0.; y[1]=6.; }
  else if( resonanceType.Contains("Upsilon") ) { y[0]=8.; y[1]=12.; }
  else if( resonanceType == "Z" ) { y[0]=80; y[1]=100; }

  // This is used to have a canvas containing both histogram points
  TGraph *gr = new TGraph(2,x,y);
  gr->SetMarkerStyle(8);
  gr->SetMarkerColor(108);
  gr->GetYaxis()->SetTitle("Res Mass(GeV)   ");
  gr->GetYaxis()->SetTitleOffset(1);
  gr->GetXaxis()->SetTitle(xAxisTitle);
  gr->SetTitle(title);

  // Text for the fits
  TPaveText * paveText1 = new TPaveText(0.20,0.15,0.49,0.28,"NDC");
  paveText1->SetFillColor(0);
  paveText1->SetTextColor(1);
  paveText1->SetTextSize(0.02);
  paveText1->SetBorderSize(1);
  TPaveText * paveText2 = new TPaveText(0.59,0.15,0.88,0.28,"NDC");
  paveText2->SetFillColor(0);
  paveText2->SetTextColor(2);
  paveText2->SetTextSize(0.02);
  paveText2->SetBorderSize(1);

  TPaveText * paveText3 = new TPaveText(0.59,0.32,0.88,0.45,"NDC");
  paveText3->SetFillColor(0);
  paveText3->SetTextColor(4);
  paveText3->SetTextSize(0.02);
  paveText3->SetBorderSize(1);
  
  /*****************PARABOLIC FIT (ETA)********************/
  if( name.Contains("Eta") ) {
    std::cout << "Fitting eta" << std::endl;
    TF1 *fit1 = new TF1("fit1",onlyParabolic,-3.2,3.2,2);
    fit1->SetLineWidth(2);
    fit1->SetLineColor(1);
    if( grM_1->GetN() > 0 ) grM_1->Fit("fit1","", "", -3, 3);
    setTPaveText(fit1, paveText1);

    TF1 *fit2 = new TF1("fit2","pol0",-3.2,3.2);
    // TF1 *fit2 = new TF1("fit2",onlyParabolic,-3.2,3.2,2);
    fit2->SetLineWidth(2);
    fit2->SetLineColor(2);
    if( grM_2->GetN() > 0 ) grM_2->Fit("fit2","", "", -3, 3);
    // grM_2->Fit("fit2","R");
    setTPaveText(fit2, paveText2);
    if(genMass){
      TF1 *fit3 = new TF1("fit3",onlyParabolic,-3.2,3.2,2);
      fit3->SetLineWidth(2);
      fit3->SetLineColor(4);
      if( grM_Gen->GetN() > 0 ) grM_Gen->Fit("fit3","", "", -3, 3);
      // grM_2->Fit("fit2","R");
      setTPaveText(fit3, paveText3);
    }
  }
  /*****************SINUSOIDAL FIT (PHI)********************/
//   if( name.Contains("Phi") ) {
//     std::cout << "Fitting phi" << std::endl;
//     TF1 *fit1 = new TF1("fit1",sinusoidal,-3.2,3.2,3);
//     fit1->SetParameters(9.45,1,1);
//     fit1->SetParNames("offset","amplitude","phase");
//     fit1->SetLineWidth(2);
//     fit1->SetLineColor(1);
//     fit1->SetParLimits(2,-3.14,3.14);
//     if( grM_1->GetN() > 0 ) grM_1->Fit("fit1","","",-3,3);
//     setTPaveText(fit1, paveText1);

//     TF1 *fit2 = new TF1("fit2",sinusoidal,-3.2,3.2,3);
//     fit2->SetParameters(9.45,1,1);
//     fit2->SetParNames("offset","amplitude","phase");
//     fit2->SetLineWidth(2);
//     fit2->SetLineColor(2);
//     fit2->SetParLimits(2,-3.14,3.14);
//     if( grM_2->GetN() > 0 ) grM_2->Fit("fit2","","",-3,3);
//     setTPaveText(fit2, paveText2);

//     if(genMass){
//       TF1 *fit3 = new TF1("fit3",sinusoidal,-3.2,3.2,3);
//       fit3->SetParameters(9.45,1,1);
//       fit3->SetParNames("offset","amplitude","phase");
//       fit3->SetLineWidth(2);
//       fit3->SetLineColor(4);
//       fit3->SetParLimits(2,-3.14,3.14);
//       if( grM_Gen->GetN() > 0 ) grM_Gen->Fit("fit3","","",-3,3);
//       setTPaveText(fit3, paveText3);
//     }
//   }
  /*****************************************/ 
  if( name.Contains("Pt") ) {
    std::cout << "Fitting pt" << std::endl;
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

    if(genMass){
      TF1 * fit3 = new TF1("fit3", "[0]", 0., 150.);
      fit3->SetParameters(0., 1.);
      fit3->SetParNames("scale","pt coefficient");
      fit3->SetLineWidth(2);
      fit3->SetLineColor(4);
      if( grM_Gen->GetN() > 0 ) {
	if( name.Contains("Z") ) grM_Gen->Fit("fit3","","",0.,150.);
	grM_Gen->Fit("fit3","","",0.,27.);
      }
      setTPaveText(fit3, paveText3);
    }
  }

  c->cd();
  gr->Draw("AP");
  grM_1->Draw("P");
  grM_2->Draw("P");
  if(genMass)
    grM_Gen->Draw("P");

  paveText1->Draw("same");
  paveText2->Draw("same");
  if(genMass)
    paveText3->Draw("same");

  TLegend *leg = new TLegend(0.65,0.85,1,1);
  leg->SetFillColor(0);
  leg->AddEntry(grM_1,"before calibration","P");
  leg->AddEntry(grM_2,"after calibration","P");
  if(genMass)
    leg->AddEntry(grM_Gen, "generated mass", "P");
  leg->Draw("same");

  outputFile->cd();
  c->Write();
  outputFile->Close();
}

Double_t sinusoidal(Double_t *x, Double_t *par) {
  return (par[0] + par[1]*sin(x[0]+par[2])); 
}

Double_t parabolic(Double_t *x, Double_t *par) {
  return (par[0] + par[1]*fabs(x[0]) + par[2]*x[0]*x[0]) ; 
}

Double_t overX(Double_t *x, Double_t *par) {
  return (par[0] + par[1]/x[0]) ; 
}

Double_t parabolic2(Double_t *x, Double_t *par) {
  if(x>0)
    return (par[0] + par[1]*fabs(x[0]) + par[2]*x[0]*x[0]) ; 
  else
    return (par[0] - par[1]*fabs(x[0]) + par[2]*x[0]*x[0]) ; 
}

Double_t onlyParabolic(Double_t *x, Double_t *par) {
  return (par[0] + par[1]*x[0]*x[0]) ; 
}

Double_t linear(Double_t *x, Double_t *par) {
  return (par[0] + par[1]*fabs(x[0])) ; 
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
