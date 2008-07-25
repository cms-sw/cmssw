#include <vector>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include "TFile.h"
#include "TTree.h"
#include "TDirectoryFile.h"
#include "TH1.h"
#include "TF1.h"
#include "TROOT.h"
#include "TCanvas.h"
#include "TPaveText.h"
#include "TDirectory.h"
#include "TMath.h"
#include "TStyle.h"
#include "DQMServices/Core/interface/MonitorElement.h"


/**
@ class fitME 
@ fit Landau distributions to historic monitoring elements
@ fits from Susy's analysis (DQM/SiStripHistoricInfoClient/test/TrendsWithFits) 
*/



Double_t langaufun(Double_t *x, Double_t *par);
Int_t langaupro(Double_t *params, Double_t &maxx, Double_t &FWHM);

class fitME{

 public: 

  fitME(MonitorElement* ME);
  ~fitME();
 
  Stat_t doFit();
  Stat_t doNoiseFit();
  
  Double_t* getFitPar() {return pLanGausS;} 
  Double_t* getNoisePar() {return pGausS;}
  Double_t* getFitParErr() {return epLanGausS;}
  Double_t* getNoiseParErr() {return epGausS;}
  Double_t  getFitChi() {return chi2GausS;}
  Int_t     getFitnDof() {return nDofGausS;}
 
 private:
  
  Double_t *pLanGausS, *epLanGausS;
  Double_t *pGausS, *epGausS;
  Double_t *pLanConv;
  Double_t chi2GausS;
  Int_t nDofGausS;
  TF1 *langausFit; 
  TF1 *gausFit;
  TH1 *htoFit; 
  
};

