
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2013/05/23 15:28:45 $
 *  $Revision: 1.10 $
 *  \author S. Bolognesi - INFN Torino
 */

#include "CalibMuon/DTCalibration/interface/DTMeanTimerFitter.h"
//#include "CalibMuon/DTCalibration/plugins/vDriftHistos.h"
#include "CalibMuon/DTCalibration/interface/vDriftHistos.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>

#include "TFile.h"
#include "TF1.h"

using namespace std;

DTMeanTimerFitter::DTMeanTimerFitter(TFile *file) : hInputFile(file), theVerbosityLevel(0) {
  //hInputFile = new TFile("DTTMaxHistos.root", "READ");
  hDebugFile = new TFile("DTMeanTimerFitter.root", "RECREATE");
}

DTMeanTimerFitter::~DTMeanTimerFitter() {
  hDebugFile->Close();
}

vector<float> DTMeanTimerFitter::evaluateVDriftAndReso (const TString& N) {
  
  // Retrieve histogram sets
  hTMaxCell * histos   = new hTMaxCell(N, hInputFile);
  vector<float> vDriftAndReso;

  // Check that the histo for this cell exists
  if(histos->hTmax123 != 0) {
    vector<TH1F*> hTMax;  // histograms for <T_max> calculation
    vector <TH1F*> hT0;   // histograms for T0 evaluation
    hTMax.push_back(histos->hTmax123); 
    hTMax.push_back(histos->hTmax124s72);
    hTMax.push_back(histos->hTmax124s78);
    hTMax.push_back(histos->hTmax134s72);
    hTMax.push_back(histos->hTmax134s78);
    hTMax.push_back(histos->hTmax234);

    hT0.push_back(histos->hTmax_3t0);
    hT0.push_back(histos->hTmax_2t0);
    hT0.push_back(histos->hTmax_t0);
    hT0.push_back(histos->hTmax_0);

    vector<Double_t> factor; // factor relating the width of the Tmax distribution 
                             // and the cell resolution 
    factor.push_back(sqrt(2./3.)); // hTmax123
    factor.push_back(sqrt(2./7.)); // hTmax124s72
    factor.push_back(sqrt(8./7.)); // hTmax124s78
    factor.push_back(sqrt(2./7.)); // hTmax134s72
    factor.push_back(sqrt(8./7.)); // hTmax134s78
    factor.push_back(sqrt(2./3.)); // hTmax234


    // Retrieve the gaussian mean and sigma for each TMax histogram    
    vector<Double_t> mean;
    vector<Double_t> sigma; 
    vector<Double_t> count;  //number of entries

    for(vector<TH1F*>::const_iterator ith = hTMax.begin();
	ith != hTMax.end(); ith++) {
      TF1 *funct = fitTMax(*ith);
      if(!funct){
	edm::LogError("DTMeanTimerFitter") << "Error when fitting TMax..histogram name" << (*ith)->GetName();
        // return empty or -1 filled vector?
        vector<float> defvec(6,-1);
        return defvec;
      }			
      hDebugFile->cd();
      (*ith)->Write();

    // Get mean, sigma and number of entries of each histogram
      mean.push_back(funct->GetParameter(1));
      sigma.push_back(funct->GetParameter(2)); 
      count.push_back((*ith)->GetEntries());  
    } 
  	  
    Double_t tMaxMean=0.;
    Double_t wTMaxSum=0.;
    Double_t sigmaT=0.;
    Double_t wSigmaSum = 0.;
  
    //calculate total mean and sigma
    for(int i=0; i<=5; i++) {
      if(count[i]<200) continue;
      tMaxMean  += mean[i]*(count[i]/(sigma[i]*sigma[i]));
      wTMaxSum  += count[i]/(sigma[i]*sigma[i]);
      sigmaT    += count[i]*factor[i]*sigma[i];
      wSigmaSum += count[i];
      // cout << "TMaxMean "<<i<<": "<< mean[i] << " entries: " << count[i] 
      // << " sigma: " << sigma[i] 
      // << " weight: " << (count[i]/(sigma[i]*sigma[i])) << endl; 
    }
    if((!wTMaxSum)||(!wSigmaSum)){
      edm::LogError("DTMeanTimerFitter") << "Error zero sum of weights..returning default";
      vector<float> defvec(6,-1);
      return defvec;	
    }

    tMaxMean /= wTMaxSum;
    sigmaT /= wSigmaSum;

    //calculate v_drift and resolution
    Double_t vDrift = 2.1 / tMaxMean; //2.1 is the half cell length in cm
    Double_t reso = vDrift * sigmaT;
    vDriftAndReso.push_back(vDrift);
    vDriftAndReso.push_back(reso);
    //if(theVerbosityLevel >= 1)
    edm::LogVerbatim("DTMeanTimerFitter") << " final TMaxMean=" << tMaxMean << " sigma= "  << sigmaT 
	   << " v_d and reso: " << vDrift << " " << reso << endl;

    // Order t0 histogram by number of entries (choose histograms with higher nr. of entries)
    map<Double_t,TH1F*> hEntries;    
    for(vector<TH1F*>::const_iterator ith = hT0.begin();
	ith != hT0.end(); ith++) {
      hEntries[(*ith)->GetEntries()] = (*ith);
    } 

    // add at the end of hT0 the two hists with the higher number of entries 
    int counter = 0;
    for(map<Double_t,TH1F*>::reverse_iterator iter = hEntries.rbegin();
 	iter != hEntries.rend(); iter++) {
      counter++;
      if (counter==1) hT0.push_back(iter->second); 
      else if (counter==2) {hT0.push_back(iter->second); break;} 
    }
    
    // Retrieve the gaussian mean and sigma of histograms for Delta(t0) evaluation   
    vector<Double_t> meanT0;
    vector<Double_t> sigmaT0; 
    vector<Double_t> countT0;

    for(vector<TH1F*>::const_iterator ith = hT0.begin();
	ith != hT0.end(); ith++) {
      try{
        (*ith)->Fit("gaus");
      } catch(std::exception){
        edm::LogError("DTMeanTimerFitter") << "Exception when fitting T0..histogram " << (*ith)->GetName();    
        // return empty or -1 filled vector?
        vector<float> defvec(6,-1);
        return defvec;
      }	
      TF1 *f1 = (*ith)->GetFunction("gaus");
      // Get mean, sigma and number of entries of the  histograms
      meanT0.push_back(f1->GetParameter(1));
      sigmaT0.push_back(f1->GetParameter(2));
      countT0.push_back((*ith)->GetEntries());
    }
    //calculate Delta(t0)
    if(hT0.size() != 6) { // check if you have all the t0 hists
      edm::LogVerbatim("DTMeanTimerFitter") << "t0 histograms = " << hT0.size();
      for(int i=1; i<=4;i++) {
	vDriftAndReso.push_back(-1);
      }
      return vDriftAndReso;
    }
    
    for(int it0=0; it0<=2; it0++) {      
      if((countT0[it0] > 200) && (countT0[it0+1] > 200)) {
	Double_t deltaT0 = meanT0[it0] - meanT0[it0+1];	
	vDriftAndReso.push_back(deltaT0);
      }  
      else
 	vDriftAndReso.push_back(999.);
    }
    //deltat0 using hists with max nr. of entries
    if((countT0[4] > 200) && (countT0[5] > 200)) {
      Double_t t0Diff = histos->GetT0Factor(hT0[4]) - histos->GetT0Factor(hT0[5]);
      Double_t deltaT0MaxEntries =  (meanT0[4] - meanT0[5])/ t0Diff;
      vDriftAndReso.push_back(deltaT0MaxEntries);
    }
    else
      vDriftAndReso.push_back(999.);
  }
  else {
    for(int i=1; i<=6; i++) { 
      // 0=vdrift, 1=reso,  2=(3deltat0-2deltat0), 3=(2deltat0-1deltat0), 
      // 4=(1deltat0-0deltat0), 5=deltat0 from hists with max entries,
      vDriftAndReso.push_back(-1);
    }
  }
  return vDriftAndReso;
}



TF1* DTMeanTimerFitter::fitTMax(TH1F* histo){
 // Find distribution peak and fit range
      Double_t peak = (((((histo->GetXaxis())->GetXmax())-((histo->GetXaxis())->GetXmin()))/histo->GetNbinsX())*
		       (histo->GetMaximumBin()))+((histo->GetXaxis())->GetXmin());
      //if(theVerbosityLevel >= 1)
      LogDebug("DTMeanTimerFitter") <<"Peak "<<peak<<" : "<<"xmax "<<((histo->GetXaxis())->GetXmax())
	    <<"            xmin "<<((histo->GetXaxis())->GetXmin())
	    <<"            nbin "<<histo->GetNbinsX()
	    <<"            bin with max "<<(histo->GetMaximumBin());
      Double_t range = 2.*histo->GetRMS(); 

      // Fit each Tmax histogram with a Gaussian in a restricted interval
      TF1 *rGaus = new TF1("rGaus","gaus",peak-range,peak+range);
      rGaus->SetMarkerSize();// just silence gcc complainining about unused var
      try{	
        histo->Fit("rGaus","R");
      } catch(std::exception){
	edm::LogError("DTMeanTimerFitter") << "Exception when fitting TMax..histogram " << histo->GetName()
	     << "   setting return function pointer to zero";   
	return 0;
      }	
      TF1 *f1 = histo->GetFunction("rGaus");
      return f1;
 }
