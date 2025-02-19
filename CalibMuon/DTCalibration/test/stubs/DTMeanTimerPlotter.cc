/*
 *  See header file for a description of this class.
 *
 *  $Date: 2012/01/12 15:20:03 $
 *  $Revision: 1.6 $
 *  \author S. Bolognesi - INFN Torino
 */

#include "DTMeanTimerPlotter.h"
//#include "CalibMuon/DTCalibration/src/vDriftHistos.h"

#include <iostream>
#include <stdio.h>
#include <sstream>
#include <string>
#include <math.h>

#include "TFile.h"
#include "TCanvas.h"
#include "TCollection.h"
#include "TSystem.h"
#include "TF1.h"
#include "TH1D.h"
#include "TLegend.h"
// #include "TProfile.h"

using namespace std;

// Constructor
DTMeanTimerPlotter::DTMeanTimerPlotter(TFile *file) : theFile(file), theVerbosityLevel(1), theRebinning(1), color(0){
}

// Destructor
DTMeanTimerPlotter::~DTMeanTimerPlotter(){}


void DTMeanTimerPlotter::plotMeanTimer(int wheel, int station, int sector, int sl, const TString& drawOptions) {
  TString histoName = getHistoNameSuffix(wheel, station, sector, sl);

  if(drawOptions.Contains("SingleDeltaT0"))
    plotHistos(plotSingleTMaxDeltaT0(histoName),histoName,drawOptions);
  else if(drawOptions.Contains("SingleFormula")) 
    plotHistos(plotSingleTMaxFormula(histoName),histoName,drawOptions);
  else {
     plotHistos(plotTotalTMax(histoName),histoName,drawOptions) ;
    cout<<"use '(fit)SingleDeltaT0' or '(fit)SingleFormula' as option to fit each TMax histo separately"<<endl;
  }
}

// Set the verbosity of the output: 0 = silent, 1 = info, 2 = debug
void DTMeanTimerPlotter::setVerbosity(unsigned int lvl) {
  theVerbosityLevel = lvl;
}

void DTMeanTimerPlotter::setRebinning(unsigned int rebin) {
  theRebinning = rebin;
}

void DTMeanTimerPlotter::resetColor() {
  color = 0;
}

TString DTMeanTimerPlotter::getHistoNameSuffix(int wheel, int station, int sector, int sl) {
 TString N=(((((TString) "TMax"+(long) wheel) +(long) station)
		  +(long) sector)+(long) sl);
  return N;
}

void DTMeanTimerPlotter::plotHistos(vector<TH1D*> hTMaxes, TString& name, const TString& drawOptions){

  TLegend *leg = new TLegend(0.5,0.6,0.7,0.8);;
  if(!drawOptions.Contains("same")){
     new TCanvas(name,"Fit of Tmax histo");
     hTMaxes[0]->Draw();
   }

  for(vector<TH1D*>::const_iterator ith = hTMaxes.begin();
      ith != hTMaxes.end(); ith++) {
   color++;     

   (*ith)->Rebin(theRebinning);
   (*ith)->SetLineColor(color);
   ((*ith)->GetXaxis())->SetRangeUser(200,600);
   (*ith)->Draw("same");

   leg->AddEntry((*ith),(*ith)->GetName(),"L");
 }

 if(drawOptions.Contains("fit")){
   int i=0;
   vector<TF1*> functions = fitTMaxes(hTMaxes);
   for(vector<TF1*>::const_iterator funct = functions.begin();
       funct != functions.end(); funct++) {
     //     color++;     
     (*funct)->SetLineColor(hTMaxes[i]->GetLineColor());
     (*funct)->Draw("same");
     i++;
   }
 }
 leg->SetFillColor(0);;
 leg->Draw("same");
 hTMaxes[0]->SetMaximum(getMaximum(hTMaxes));
 setRebinning(1);
}

vector<TH1D*> DTMeanTimerPlotter::plotSingleTMaxDeltaT0 (TString& name) {
 // Retrieve histogram sets
   vector <TH1D*> hTMaxes;
  hTMaxes.push_back((TH1D*)theFile->Get(name+"_0"));
  hTMaxes.push_back((TH1D*)theFile->Get(name+"_t0"));
  hTMaxes.push_back((TH1D*)theFile->Get(name+"_2t0"));
  hTMaxes.push_back((TH1D*)theFile->Get(name+"_3t0"));
  if(theVerbosityLevel >= 1){
    cout<<"NOTE: these histos are not directly used to calibrate vdrift"<<endl;
  }
  return hTMaxes;
}

vector<TH1D*>  DTMeanTimerPlotter::plotSingleTMaxFormula (TString& name) {
 // Retrieve histogram sets
   vector <TH1D*> hTMaxes;
  hTMaxes.push_back((TH1D*)theFile->Get(name+"_Tmax123"));
  hTMaxes.push_back((TH1D*)theFile->Get(name+"_Tmax124_s72"));
  hTMaxes.push_back((TH1D*)theFile->Get(name+"_Tmax124_s78"));
  hTMaxes.push_back((TH1D*)theFile->Get(name+"_Tmax134_s72"));
  hTMaxes.push_back((TH1D*)theFile->Get(name+"_Tmax134_s78"));
  hTMaxes.push_back((TH1D*)theFile->Get(name+"_Tmax234"));
  return hTMaxes;
}  

vector<TH1D*>  DTMeanTimerPlotter::plotTotalTMax (TString& name) {

  TH1D* hTMax;  // histograms for <T_max> calculation
  hTMax = ((TH1D*)theFile->Get(name+"_Tmax123"));
  hTMax->Add((TH1D*)theFile->Get(name+"_Tmax124_s72"));
  hTMax->Add((TH1D*)theFile->Get(name+"_Tmax124_s78"));
  hTMax->Add((TH1D*)theFile->Get(name+"_Tmax134_s72"));
  hTMax->Add((TH1D*)theFile->Get(name+"_Tmax134_s78"));
  hTMax->Add((TH1D*)theFile->Get(name+"_Tmax234"));
  hTMax->SetName(name);
  hTMax->SetTitle(name);

  vector <TH1D*> hTMaxes;
  hTMaxes.push_back(hTMax);
  if(theVerbosityLevel >= 1){
    cout<<"NOTE: this histo is not directly used to calibrate vdrift"<<endl;
 }
  return hTMaxes;
}

vector<TF1*>  DTMeanTimerPlotter::fitTMaxes(vector<TH1D*> hTMaxes){
  vector <TF1*> functions;
   for(vector<TH1D*>::const_iterator ith = hTMaxes.begin();
      ith != hTMaxes.end(); ith++) {
     // Find distribution peak and fit range
      Double_t peak = ((((((*ith)->GetXaxis())->GetXmax())-(((*ith)->GetXaxis())->GetXmin()))/(*ith)->GetNbinsX())*
		       ((*ith)->GetMaximumBin()))+(((*ith)->GetXaxis())->GetXmin());
      if(theVerbosityLevel >= 1)
	cout<<"Peak "<<peak<<" : "<<"xmax "<<(((*ith)->GetXaxis())->GetXmax())
	    <<"            xmin "<<(((*ith)->GetXaxis())->GetXmin())
	    <<"            nbin "<<(*ith)->GetNbinsX()
	    <<"            bin with max "<<((*ith)->GetMaximumBin())<<endl;
      Double_t range = 2.*(*ith)->GetRMS(); 

      // Fit each Tmax (*ith)gram with a Gaussian in a restricted interval
      TF1 *rGaus = new TF1("rGaus","gaus",peak-range,peak+range);
      rGaus->SetMarkerSize(); //to stop gcc complain about unused var
      (*ith)->Fit("rGaus","R0");
      functions.push_back((*ith)->GetFunction("rGaus"));
      
      // Get mean, sigma and number of entries of each histogram
      cout<<"Histo name "<<(*ith)->GetName()<<": "<<endl;
      cout<<"mean  "<<(((*ith)->GetFunction("rGaus"))->GetParameter(1))<<endl;
      cout<<"sigma "<<(((*ith)->GetFunction("rGaus"))->GetParameter(2))<<endl; 
      cout<<"count "<<((*ith)->GetEntries())<<endl<<endl;  
   }
   return functions;
 }

double DTMeanTimerPlotter::getMaximum(vector<TH1D*> hTMaxes){
  double max = -pow(10.0,10);
  for(vector<TH1D*>::const_iterator ith = hTMaxes.begin();
      ith != hTMaxes.end(); ith++) {
    double m =(*ith)->GetMaximum(pow(10.0,10));
    if(m>max)
      max = m;
  }
  return max;
}
