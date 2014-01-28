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

bool debug = true;
using namespace std;

Double_t langaufun(Double_t *x, Double_t *par);
Int_t langaupro(Double_t *params, Double_t &maxx, Double_t &FWHM);

//---------------- Global Functions ------------------//
Double_t langaufun(Double_t *x, Double_t *par) {
   //Fit parameters:
   //par[0]=Width (scale) parameter of Landau density
   //par[1]=Most Probable (MP, location) parameter of Landau density
   //par[2]=Total area (integral -inf to inf, normalization constant)
   //par[3]=Width (sigma) of convoluted Gaussian function
   //
   //In the Landau distribution (represented by the CERNLIB approximation), 
   //the maximum is located at x=-0.22278298 with the location parameter=0.
   //This shift is corrected within this function, so that the actual
   //maximum is identical to the MP parameter.

      // Numeric constants
      Double_t invsq2pi = 0.3989422804014;   // (2 pi)^(-1/2)
      Double_t mpshift  = -0.22278298;       // Landau maximum location

      // Control constants
      Double_t np = 100.0;      // number of convolution steps
      Double_t sc =   5.0;      // convolution extends to +-sc Gaussian sigmas

      // Variables
      Double_t xx;
      Double_t mpc;
      Double_t fland;
      Double_t sum = 0.0;
      Double_t xlow,xupp;
      Double_t step;
      Double_t i;


      // MP shift correction
      mpc = par[1] - mpshift * par[0]; 

      // Range of convolution integral
      xlow = x[0] - sc * par[3];
      xupp = x[0] + sc * par[3];

      step = (xupp-xlow) / np;

      // Landau Distribution Production
      for(i=1.0; i<=np/2; i++) {
         xx = xlow + (i-.5) * step;
         fland = TMath::Landau(xx,mpc,par[0]) / par[0];
         sum += fland * TMath::Gaus(x[0],xx,par[3]);

         xx = xupp - (i-.5) * step;
         fland = TMath::Landau(xx,mpc,par[0]) / par[0];
         sum += fland * TMath::Gaus(x[0],xx,par[3]);
      }

      return (par[2] * step * sum * invsq2pi / par[3]);
}

Int_t langaupro(Double_t *params, Double_t &maxx, Double_t &FWHM) {
  cout << "inside langaupro " << endl;
   // Seaches for the location (x value) at the maximum of the 
   // Landau and its full width at half-maximum.
   //
   // The search is probably not very efficient, but it's a first try.

   Double_t p,x,fy,fxr,fxl;
   Double_t step;
   Double_t l,lold,dl;
   Int_t i = 0;
   const Int_t MAXCALLS = 10000;
   const Double_t dlStop = 1e-3; // relative change < .001

   // Search for maximum
   p = params[1] - 0.1 * params[0];
   step = 0.05 * params[0];
   lold = -2.0;
   l    = -1.0;

   dl = (l-lold)/lold;    // FIXME catch divide by zero
   while ( (TMath::Abs(dl)>dlStop ) && (i < MAXCALLS) ) {
      i++;

      lold = l;
      x = p + step;
      l = langaufun(&x,params);
      dl = (l-lold)/lold; // FIXME catch divide by zero
        
      if (l < lold)
         step = -step/10;
 
      p += step;
   }

   if (i == MAXCALLS)
      return (-1);

   maxx = x;

   fy = l/2;


   // Search for right x location of fy
   p = maxx + params[0];
   step = params[0];
   lold = -2.0;
   l    = -1e300;
   i    = 0;

   dl = (l-lold)/lold;   // FIXME catch divide by zero
   while ( ( TMath::Abs(dl)>dlStop ) && (i < MAXCALLS) ) {
      i++;

      lold = l;
      x = p + step;
      l = TMath::Abs(langaufun(&x,params) - fy);
      dl = (l-lold)/lold; // FIXME catch divide by zero
 
      if (l > lold)
         step = -step/10;
 
      p += step;
   }

   if (i == MAXCALLS)
      return (-2);

   fxr = x;


   // Search for left x location of fy
   p = maxx - 0.5 * params[0];
   step = -params[0];
   lold = -2.0;
   l    = -1e300;
   i    = 0;

   dl = (l-lold)/lold;    // FIXME catch divide by zero
   while ( ( TMath::Abs(dl)>dlStop ) && (i < MAXCALLS) ) {
      i++;

      lold = l;
      x = p + step;
      l = TMath::Abs(langaufun(&x,params) - fy);
      dl = (l-lold)/lold; // FIXME catch divide by zero
 
      if (l > lold)
         step = -step/10;
 
      p += step;
   }

   if (i == MAXCALLS)
      return (-3);


   fxl = x;

   FWHM = fxr - fxl;
   return (0);
}
Double_t Gauss(Double_t *x, Double_t *par)
// The noise function: a gaussian
{
   Double_t arg = 0;
   if (par[2]) arg = (x[0] - par[1])/par[2];

   Double_t noise = par[0]*TMath::Exp(-0.5*arg*arg);
   return noise;
}

//----------------------------------------//
class AnalysisClass{

 public: 
  //  AnalysisClass(vector<string>::iterator it);  
  AnalysisClass(const char*,Int_t r);
  ~AnalysisClass();

 //====== Histo ======//
  Stat_t doHisto(Char_t*,Char_t*);
  Double_t *getHistoPar() {return pPar;} 
  Stat_t doGlobalHisto(Char_t*);
  //====== Fit ======//
  Stat_t doFit(Int_t, Char_t*, Char_t* );
  Stat_t doNoiseFit(Int_t, Char_t*, Char_t*);
  Double_t* getFitPar() {return pLanGausS;} 
  Double_t *getNoisePar() {return pGausS;}
  Double_t* getFitParErr() {return epLanGausS;}
  Double_t* getNoiseParErr() {return epGausS;}
  Double_t getFitChi() {return chi2GausS;}
  Int_t getFitnDof() {return nDofGausS;}

 private:
  Int_t fRun;
  TFile *fName;
  Double_t *pPar;
  Double_t *pLanGausS, *epLanGausS;
  Double_t *pGausS, *epGausS;
  Double_t *pLanConv;
  Double_t chi2GausS;
  Int_t nDofGausS;
  TF1 *langausFit; 
  TF1 *gausFit;
  TH1 *h; 
  TDirectoryFile*  d1;
  TDirectoryFile*  dir;
};

AnalysisClass::AnalysisClass(const char* RootFileName, Int_t r) : fRun(r) {
  cout << "fRun " << fRun << endl;
  Char_t RunName[10];
  sprintf(RunName,"%d",r);
  cout << "RunName " << RunName << endl;
  Char_t RootFileName_[100];
  sprintf(RootFileName_,RootFileName);
  fName=new TFile(RootFileName_);
  if(fName) cout << "File open" << fName->GetPath() << endl;
  Char_t DirName[100];
  sprintf(DirName,"DQMData/Run %s/SiStrip/Run summary/MechanicalView", RunName);  
  d1 = (TDirectoryFile * )((TDirectoryFile *)fName->Get(DirName)); 
  Char_t TrackDir[100];
  sprintf(TrackDir,"DQMData/Run %s/SiStrip/Run summary/Tracks", RunName);
  dir = (TDirectoryFile * )((TDirectoryFile *)fName->Get(TrackDir)); 
  if(!d1) cout << "Directory " << DirName << " not found!" << endl;
  if(!dir) cout << "Directory " << TrackDir << " not found!" << endl;
  pPar=new Double_t[3];
  pLanGausS=new Double_t[4];
  pGausS=new Double_t[3];
  epGausS=new Double_t[3];
  epLanGausS=new Double_t[4];
  pLanConv=new Double_t[2];  
  langausFit=0;
  gausFit=0;
  cout << "end of the constructor " << endl;
}

AnalysisClass::~AnalysisClass(){
  if ( fName!=0 ) delete  fName;
  if ( pPar!=0 ) delete [] pPar;
  if ( pLanGausS!=0 ) delete [] pLanGausS;
  if ( epLanGausS!=0 ) delete [] epLanGausS;
  if ( pGausS!=0 ) delete [] pGausS;
  if ( epGausS!=0 ) delete [] epGausS;
  if ( pLanConv!=0) delete[] pLanConv;
  if ( langausFit!=0 ) delete langausFit;
  if ( gausFit!=0 ) delete gausFit;
  if ( d1 ) delete d1;
  if ( dir ) delete dir;
}

Stat_t AnalysisClass::doHisto(Char_t * Variable, Char_t *SubDetName){
  TH1 *hHisto=0; 
  if(debug)  cout << d1->GetTitle() << " " << Variable << " " << SubDetName << endl;
  pPar[0]=0; pPar[1]=0;
  TIter it(d1->GetListOfKeys());
  TObject * o;
  while ( (o = it()))  
    {
      TObject * d =  d1->Get(o->GetName());
      if(d->IsA()->InheritsFrom("TDirectory") && strstr(d->GetName(),SubDetName)){
	if(d){
	  if (debug) cout << "Found " << SubDetName << endl;
	  TIter it2(((TDirectoryFile * )d)->GetListOfKeys());
	  TObject *o2;
	  while( ( o2 = it2()) ){
	    TObject *d2 = ((TDirectoryFile * )d)->Get(o2->GetName());
	    if(strstr(d2->GetName(),Variable)){
	      if(debug) cout << "Found " << Variable << endl;
	      hHisto = (TH1*) d2;
	      if(hHisto->GetEntries()!=0) {
		pPar[0]=hHisto->GetMean();
		pPar[1]=hHisto->GetRMS();  
		pPar[2]=hHisto->GetEntries()-hHisto->GetBinContent(1);
		if (debug) cout << "Histo Title " << hHisto->GetTitle() << " mean: " << hHisto->GetMean() << " rms: " << hHisto->GetRMS() << " " << hHisto->GetEntries() << endl;
	      }
	      else{
		cout<<"Empty Histogram "<< hHisto->GetTitle() << endl;
		pPar[0]=-10; pPar[1]=-10;
	      }
	    }
	  }
	}
      }
    }
  return hHisto->GetEntries();
  
}

Stat_t AnalysisClass::doGlobalHisto(Char_t * Variable){

  pPar[0]=0; pPar[1]=0; pPar[2]=0;
  TH1 *h=0; 
  if(debug)  cout << d1->GetTitle() << " " << Variable << endl;
  pPar[0]=0; pPar[1]=0;
  TIter it(dir->GetListOfKeys());
  TObject * o;
  while ( (o = it()))  
    {
      TObject * d =  dir->Get(o->GetName());
      if(d->IsA()->InheritsFrom("TH1") && strstr(d->GetName(),Variable) && !strstr(d->GetName(),"Trend")){
	if(debug) cout << "Found " << d->GetName() << endl;
	if(d){
	  h= (TH1*) d;
	  if(h->GetEntries()!=0) {
	    pPar[0]=h->GetMean();
	    pPar[1]=h->GetRMS(); 
	    pPar[2]=h->GetEntries()-h->GetBinContent(1);  
	    if (debug) cout << "Histo Title " << h->GetTitle() << " mean: " << h->GetMean() << " non zero entries: " << pPar[2] << endl;
	  }
	  else{
	    cout<<"Empty Histogram "<< h->GetTitle() << endl;
	    pPar[0]=-10; pPar[1]=-10; pPar[2]=-10;
	  }
	}
      }
    }
  return h->GetEntries();
}

Stat_t AnalysisClass::doFit(Int_t RunNumber,Char_t *Variable, Char_t *SubDetName){
  TH1 *htoFit=0; 
  pLanGausS[0]=0; pLanGausS[1]=0; pLanGausS[2]=0; pLanGausS[3]=0;
  epLanGausS[0]=0; epLanGausS[1]=0; epLanGausS[2]=0; epLanGausS[3]=0;

  if (debug) cout << d1->GetTitle() << " " << Variable << " " << SubDetName << endl;
  pPar[0]=0; pPar[1]=0;
  TIter it(d1->GetListOfKeys());
  TObject * o;
  while ( (o = it()))  
    {
      TObject * d =  d1->Get(o->GetName());
      if(d->IsA()->InheritsFrom("TDirectory") && strstr(d->GetName(),SubDetName)){
	if (debug) cout << "Found " << SubDetName << endl;
	TIter it2(((TDirectoryFile * )d)->GetListOfKeys());
	TObject *o2;
	while( ( o2 = it2()) ){
	  TObject *d2 = ((TDirectoryFile * )d)->Get(o2->GetName());
	  if(strstr(d2->GetName(),Variable)){
	    if (debug) cout << "Found " << Variable << endl;
	    htoFit = (TH1*) d2;

	    if (htoFit->GetEntries()!=0) {
	      cout<<"Fitting "<< htoFit->GetTitle() <<endl;
	      // Setting fit range and start values
	      Double_t fr[2];
	      Double_t sv[4], pllo[4], plhi[4];
	      fr[0]=0.5*htoFit->GetMean();
	      fr[1]=3.0*htoFit->GetMean();
	      
	      // (EM) parameters setting good for signal only 
	      Int_t imax=htoFit->GetMaximumBin();
	      Double_t xmax=htoFit->GetBinCenter(imax);
	      Double_t ymax=htoFit->GetBinContent(imax);
	      Int_t i[2];
	      Int_t iArea[2];
	      
	      i[0]=htoFit->GetXaxis()->FindBin(fr[0]);
	      i[1]=htoFit->GetXaxis()->FindBin(fr[1]);
	      
	      iArea[0]=htoFit->GetXaxis()->FindBin(fr[0]);
	      iArea[1]=htoFit->GetXaxis()->FindBin(fr[1]);
	      Double_t AreaFWHM=htoFit->Integral(iArea[0],iArea[1],"width");
	      
	      sv[1]=xmax;
	      sv[2]=htoFit->Integral(i[0],i[1],"width");
	      sv[3]=AreaFWHM/(4*ymax);
	      sv[0]=sv[3];
	      
	      plhi[0]=25.0; plhi[1]=200.0; plhi[2]=1000000.0; plhi[3]=50.0;
	      pllo[0]=1.5 ; pllo[1]=10.0 ; pllo[2]=1.0      ; pllo[3]= 1.0;
	      
	      // create different landau+gaussians for different runs
	      Char_t FunName[100];
	      sprintf(FunName,"FitfcnLG_%s%d",htoFit->GetName(),fRun);  
	      TF1 *ffitold = (TF1*)gROOT->GetListOfFunctions()->FindObject(FunName);
	      if (ffitold) delete ffitold;
	      
	      langausFit = new TF1(FunName,langaufun,fr[0],fr[1],4);
	      langausFit->SetParameters(sv);
	      langausFit->SetParNames("Width","MP","Area","GSigma");
	      
	      for (Int_t i=0; i<4; i++) {
		langausFit->SetParLimits(i,pllo[i],plhi[i]);
	      }  
	      
	      htoFit->Fit(langausFit,"R0");  // "R" fit in a range,"0" quiet fit
	      
	      langausFit->SetRange(fr[0],fr[1]);
	      pLanGausS=langausFit->GetParameters();
	      epLanGausS=langausFit->GetParErrors();
	      if (debug) cout << "Par Err " << epLanGausS[0] << " " <<  epLanGausS[1] << " " << epLanGausS[2] << " " <<  endl;	      
	      chi2GausS =langausFit->GetChisquare();  // obtain chi^2
	      nDofGausS = langausFit->GetNDF();           // obtain ndf
	      if (debug) cout << "chi2 " << chi2GausS << endl;	      
	      Double_t sPeak, sFWHM;
	      langaupro(pLanGausS,sPeak,sFWHM);
	      pLanConv[0]=sPeak;
	      pLanConv[1]=sFWHM;
	      cout << "langaupro:  max  " << sPeak << endl;
	      cout << "langaupro:  FWHM " << sFWHM << endl;
	      
	      TCanvas *cAll = new TCanvas("Fit",htoFit->GetTitle(),1);
	      Char_t fitFileName[60];
	      sprintf(fitFileName,"Fits/Run_%d/%s/Fit_%s.png",RunNumber,SubDetName,htoFit->GetTitle());
	      htoFit->Draw("pe");
	      htoFit->SetStats(100);
	      langausFit->Draw("lsame");
	      gStyle->SetOptFit(1111111);
	      
	      cAll->Print(fitFileName,"png");
	    }
	    else {  
	      pLanGausS[0]=-10; pLanGausS[1]=-10; pLanGausS[2]=-10; pLanGausS[3]=-10;
	      epLanGausS[0]=-10; epLanGausS[1]=-10; epLanGausS[2]=-10; epLanGausS[3]=-10;
	      pLanConv[0]=-10;   pLanConv[1]=-10;   chi2GausS=-10;  nDofGausS=-10;    
	    }
	  }
	}
      }
    }

  return htoFit->GetEntries();
  
}

//Noise section
Stat_t AnalysisClass::doNoiseFit(Int_t RunNumber, Char_t *Variable, Char_t *SubDetName){
  TH1 *hNtoFit=0;
  if (debug) cout << d1->GetTitle() << " " << Variable << " " << SubDetName << endl;
  pGausS[0]=0; pGausS[1]=0; pGausS[2]=0;
  epGausS[0]=0; epGausS[1]=0; epGausS[2]=0;
  TIter it(d1->GetListOfKeys());
  TObject * o;
  while ( (o = it()))  
    {
      TObject * d =  d1->Get(o->GetName());
      if(d->IsA()->InheritsFrom("TDirectory") && strstr(d->GetName(),SubDetName)){
	if (debug) cout << "Found " << SubDetName << endl;
	TIter it2(((TDirectoryFile * )d)->GetListOfKeys());
	TObject *o2;
	while( ( o2 = it2()) ){
	  TObject *d2 = ((TDirectoryFile * )d)->Get(o2->GetName());
	  if(strstr(d2->GetName(),Variable)){
	    hNtoFit = (TH1*) d2;
	    if (debug) cout << "Found " << Variable << endl;
	    if (hNtoFit->GetEntries()!=0) {
	      if (debug)  cout<<"Fitting "<< hNtoFit->GetTitle() <<endl;
	      // Setting fit range and start values
	      Double_t fr[2];
	      Double_t sv[3], pllo[3], plhi[3];
	      fr[0]=hNtoFit->GetMean()-5*hNtoFit->GetRMS();
	      fr[1]=hNtoFit->GetMean()+5*hNtoFit->GetRMS();
	      
	      Int_t imax=hNtoFit->GetMaximumBin();
	      Double_t xmax=hNtoFit->GetBinCenter(imax);
	      Double_t ymax=hNtoFit->GetBinContent(imax);
	      Int_t i[2];
	      Int_t iArea[2];
	      
	      i[0]=hNtoFit->GetXaxis()->FindBin(fr[0]);
	      i[1]=hNtoFit->GetXaxis()->FindBin(fr[1]);
	      
	      iArea[0]=hNtoFit->GetXaxis()->FindBin(fr[0]);
	      iArea[1]=hNtoFit->GetXaxis()->FindBin(fr[1]);
	      Double_t AreaFWHM=hNtoFit->Integral(iArea[0],iArea[1],"width");
	      
	      sv[2]=AreaFWHM/(4*ymax);
	      sv[1]=xmax;
	      sv[0]=hNtoFit->Integral(i[0],i[1],"width");
	      
	      plhi[0]=1000000.0; plhi[1]=10.0; plhi[2]=10.;
	      pllo[0]=1.5 ; pllo[1]=0.1; pllo[2]=0.3;
	      Char_t FunName[100];
	      sprintf(FunName,"FitfcnLG_%s%d",hNtoFit->GetName(),fRun);
	      TF1 *ffitold = (TF1*)gROOT->GetListOfFunctions()->FindObject(FunName);
	      if (ffitold) delete ffitold;
	      
	      gausFit = new TF1(FunName,Gauss,fr[0],fr[1],3);
	      gausFit->SetParameters(sv);
	      gausFit->SetParNames("Constant","GaussPeak","Sigma");
	      
	      for (Int_t i=0; i<3; i++) {
		gausFit->SetParLimits(i,pllo[i],plhi[i]);
	      }
	      hNtoFit->Fit(gausFit,"R0");
	      
	      gausFit->SetRange(fr[0],fr[1]);
	      pGausS=gausFit->GetParameters();
	      epGausS=gausFit->GetParErrors();
	      
	      chi2GausS =gausFit->GetChisquare(); // obtain chi^2
	      nDofGausS = gausFit->GetNDF();// obtain ndf

	      TCanvas *cAllN = new TCanvas("NoiseFit",hNtoFit->GetTitle(),1);
	      Char_t fitFileName[60];
	      sprintf(fitFileName,"Fits/Run_%d/%s/Fit_%s.png",RunNumber,SubDetName,hNtoFit->GetTitle());
	      hNtoFit->Draw("pe");
	      gStyle->SetOptFit(1111111);
	      gausFit->Draw("lsame");
	      
	      cAllN->Print(fitFileName,"png");
	    }else {
	      pGausS[0]=-10; pGausS[1]=-10; pGausS[2]=-10;
	      epGausS[0]=-10; epGausS[1]=-10; epGausS[2]=-10;
	    }
	  }
	}
      }
    }
  return hNtoFit->GetEntries();
}
	    
//---------- Summary Tree --------------// 

struct RunInfo_t {
  Int_t number;
  // maybe add here latency, APV mode...
};

struct HistoValues_t { 
  Stat_t entries;
  Double_t mean; 
  Double_t rms;
  Double_t nonzero;
}; 

struct LanGausValues_t { 
  Stat_t entries;
  Double_t width, ewidth;
  Double_t mp, emp;
  Double_t area, earea;
  Double_t gsigma,  egsigma;
  Double_t peak, FWHM;
  Double_t mp_peak;
  Double_t chi2red;
}; 

struct GausValues_t {
  Stat_t entries;
  Double_t garea, egarea;
  Double_t fitmean, efitmean;
  Double_t fitrms, efitrms;

};

struct SubDetSummary_t {
  HistoValues_t HistoPar;
  LanGausValues_t FitPar;
  GausValues_t FitNoisePar;
};

struct TotSummary_t {
  SubDetSummary_t On;
  SubDetSummary_t Off;
};

struct FinalSummary_t {
  TotSummary_t Charge;
  TotSummary_t ChargeCorr;
  TotSummary_t StoN;
  TotSummary_t StoNCorr;
  TotSummary_t Noise;
  TotSummary_t Width;
  TotSummary_t nClusters;
};

struct ClusAndTracks_t {
  HistoValues_t nTracks;
  HistoValues_t OnClus;
  HistoValues_t OffClus;
};

//============== > MAIN < ================//
void SubDetTree(string fileNameList, char* outputFile) {

  fstream *infile;
  string _buff;

  infile=new fstream(fileNameList.c_str(), ios::in);

  if(!infile)
    cout << "Unable to open the file " << fileNameList << endl;
 
  vector<int> *RunNumb = new vector<int>;
  vector<string> *FileName = new vector<string>; 

  int numRun;
  string fileName_;

  while(getline(*infile, _buff,'\n') ){
    stringstream os;
    os<<_buff;
    os>>numRun>>fileName_;

    RunNumb->push_back(numRun);
    FileName->push_back(fileName_);
  }
  cout << "DONE" << endl;   

  // tree leaves
  RunInfo_t theRun;
  FinalSummary_t* sdsDet[4];
  ClusAndTracks_t *ClusAndTracks;
  ClusAndTracks = new ClusAndTracks_t;
 
  for(Int_t iDet=0; iDet<4;iDet++) {
    sdsDet[iDet]=new FinalSummary_t;
  }
  
  if(debug)
    cout<<"TTree to be initialized"<<endl;
  // tree declarations
  TFile *f=new TFile(outputFile,"RECREATE");
  TTree *tree=new TTree("DataTree","Summary Tree for Global Run data");
  tree->Branch("Run", &theRun,"number/I");
  tree->Branch("TrAndClus.","ClusAndTracks_t",&ClusAndTracks, 32000, 1);
  tree->Branch("TIB.", "FinalSummary_t",&(sdsDet[0]), 32000, 2);
  tree->Branch("TOB.", "FinalSummary_t",&(sdsDet[1]), 32000, 2);
  tree->Branch("TID.", "FinalSummary_t",&(sdsDet[2]), 32000, 2);
  tree->Branch("TEC.", "FinalSummary_t",&(sdsDet[3]), 32000, 2); 
  if(debug)
    cout<<"TTree initialized"<<endl;
  
  
  Char_t *DetList[4] = {"TIB","TOB","TID","TEC"};
  AnalysisClass *aRun;
  
  // loop on the runs

  vector<int>::iterator it = RunNumb->begin();
  for(unsigned int count = 0; it != RunNumb->end(); ++it, ++count){
    
    const char* FileName_;
    FileName_ = ((*FileName)[count]).c_str();
    Int_t nRun;
    nRun = (*RunNumb)[count];
    
    cout<<"RunNumbInt "<< nRun <<"  "<<"File = "<<FileName_<<endl;
    //for (Int_t iRun=0; iRun<nRun; ++iRun){    
    
    aRun = new AnalysisClass(FileName_,nRun);
    cout << "Class initialized" << endl;
    theRun.number=nRun;
    cout << "=================> RUN NUMBER: " << theRun.number <<endl;
    
      //========== Tracks anc Clusters HISTO======//
    ClusAndTracks->nTracks.entries=aRun->doGlobalHisto("NumberOfTracks_CosmicTk");
    cout << "Number of Processed Events in Run " << theRun.number << ": " << ClusAndTracks->nTracks.entries << endl;
    ClusAndTracks->nTracks.mean=(aRun->getHistoPar())[0];
    ClusAndTracks->nTracks.nonzero=(aRun->getHistoPar())[2];
    
    ClusAndTracks->OnClus.entries=aRun->doGlobalHisto("OnTrack_NumberOfClusters");
    ClusAndTracks->OnClus.mean=(aRun->getHistoPar())[0];
    ClusAndTracks->OnClus.nonzero=(aRun->getHistoPar())[2];
    
    ClusAndTracks->OffClus.entries=aRun->doGlobalHisto("OffTrack_NumberOfClusters");
    ClusAndTracks->OffClus.mean=(aRun->getHistoPar())[0];
    ClusAndTracks->OffClus.nonzero=(aRun->getHistoPar())[2];    
    
    for(Int_t iDet=0; iDet<3;++iDet) {
      if(debug)
        cout<<"LOOP on Det"<<endl;

      sdsDet[iDet]->nClusters.On.HistoPar.entries=aRun->doHisto("Summary_NumberOfClusters_OnTrack_in",DetList[iDet]);
      sdsDet[iDet]->nClusters.On.HistoPar.mean=(aRun->getHistoPar())[0];
      sdsDet[iDet]->nClusters.On.HistoPar.nonzero=(aRun->getHistoPar())[2];

      sdsDet[iDet]->nClusters.Off.HistoPar.entries=aRun->doHisto("Summary_NumberOfClusters_OffTrack_in",DetList[iDet]);
      sdsDet[iDet]->nClusters.Off.HistoPar.mean=(aRun->getHistoPar())[0];
      sdsDet[iDet]->nClusters.Off.HistoPar.nonzero=(aRun->getHistoPar())[2];
      
      //==========NOISE===========//
      sdsDet[iDet]->Noise.On.FitNoisePar.entries=aRun->doNoiseFit(theRun.number,"Summary_ClusterNoise_OnTrack_in",DetList[iDet]);
      sdsDet[iDet]->Noise.On.FitNoisePar.fitmean=(aRun->getNoisePar())[1];
      sdsDet[iDet]->Noise.On.FitNoisePar.fitrms=(aRun->getNoisePar())[2];
      
      sdsDet[iDet]->Noise.On.FitNoisePar.efitmean=(aRun->getNoiseParErr())[1];
      sdsDet[iDet]->Noise.On.FitNoisePar.efitrms=(aRun->getNoiseParErr())[2];
      
      sdsDet[iDet]->Noise.Off.HistoPar.entries=aRun->doHisto("Summary_ClusterNoise_OffTrack_in",DetList[iDet]);
      sdsDet[iDet]->Noise.Off.HistoPar.mean=(aRun->getHistoPar())[0];
      sdsDet[iDet]->Noise.Off.HistoPar.rms=(aRun->getHistoPar())[1];
      
      //==========WIDTH===========//
      sdsDet[iDet]->Width.On.HistoPar.entries=aRun->doHisto("Summary_ClusterWidth_OnTrack_in",DetList[iDet]);
      sdsDet[iDet]->Width.On.HistoPar.mean=(aRun->getHistoPar())[0];
      sdsDet[iDet]->Width.On.HistoPar.rms=(aRun->getHistoPar())[1];
 
      sdsDet[iDet]->Width.Off.HistoPar.entries=aRun->doHisto("Summary_ClusterWidth_OffTrack_in",DetList[iDet]);
      sdsDet[iDet]->Width.Off.HistoPar.mean=(aRun->getHistoPar())[0];
      sdsDet[iDet]->Width.Off.HistoPar.rms=(aRun->getHistoPar())[1];

      //==========CHARGE===========//
      sdsDet[iDet]->Charge.On.FitPar.entries=aRun->doFit(theRun.number,"Summary_ClusterCharge_OnTrack_in",DetList[iDet]);
      sdsDet[iDet]->Charge.On.FitPar.width=(aRun->getFitPar())[0];
      sdsDet[iDet]->Charge.On.FitPar.mp=(aRun->getFitPar())[1];
      sdsDet[iDet]->Charge.On.FitPar.area=(aRun->getFitPar())[2];
      sdsDet[iDet]->Charge.On.FitPar.gsigma=(aRun->getFitPar())[3]; 
      sdsDet[iDet]->Charge.On.FitPar.peak=aRun->pLanConv[0];
      sdsDet[iDet]->Charge.On.FitPar.FWHM=aRun->pLanConv[1];
      sdsDet[iDet]->Charge.On.FitPar.mp_peak=(sdsDet[iDet]->Charge.On.FitPar.mp - sdsDet[iDet]->Charge.On.FitPar.peak);
      sdsDet[iDet]->Charge.On.FitPar.chi2red=aRun->getFitChi()/aRun->getFitnDof(); 

      sdsDet[iDet]->Charge.On.FitPar.ewidth=(aRun->getFitParErr())[0];
      sdsDet[iDet]->Charge.On.FitPar.emp=(aRun->getFitParErr())[1];
      sdsDet[iDet]->Charge.On.FitPar.earea=(aRun->getFitParErr())[2];
      sdsDet[iDet]->Charge.On.FitPar.egsigma=(aRun->getFitParErr())[3]; 
      
      sdsDet[iDet]->Charge.Off.HistoPar.entries=aRun->doHisto("Summary_ClusterCharge_OnTrack_in",DetList[iDet]);
      sdsDet[iDet]->Charge.Off.HistoPar.mean=(aRun->getHistoPar())[0];
      sdsDet[iDet]->Charge.Off.HistoPar.rms=(aRun->getHistoPar())[1];
     
      sdsDet[iDet]->ChargeCorr.On.FitPar.entries=aRun->doFit(theRun.number,"Summary_ClusterChargeCorr_OnTrack_in",DetList[iDet]);
      sdsDet[iDet]->ChargeCorr.On.FitPar.width=(aRun->getFitPar())[0];
      sdsDet[iDet]->ChargeCorr.On.FitPar.mp=(aRun->getFitPar())[1];
      sdsDet[iDet]->ChargeCorr.On.FitPar.area=(aRun->getFitPar())[2];
      sdsDet[iDet]->ChargeCorr.On.FitPar.gsigma=(aRun->getFitPar())[3]; 
      sdsDet[iDet]->ChargeCorr.On.FitPar.peak=aRun->pLanConv[0];
      sdsDet[iDet]->ChargeCorr.On.FitPar.FWHM=aRun->pLanConv[1];
      sdsDet[iDet]->ChargeCorr.On.FitPar.mp_peak=(sdsDet[iDet]->ChargeCorr.On.FitPar.mp - sdsDet[iDet]->ChargeCorr.On.FitPar.peak);
      sdsDet[iDet]->ChargeCorr.On.FitPar.chi2red=aRun->getFitChi()/aRun->getFitnDof(); 

      sdsDet[iDet]->ChargeCorr.On.FitPar.ewidth=(aRun->getFitParErr())[0];
      sdsDet[iDet]->ChargeCorr.On.FitPar.emp=(aRun->getFitParErr())[1];
      sdsDet[iDet]->ChargeCorr.On.FitPar.earea=(aRun->getFitParErr())[2];
      sdsDet[iDet]->ChargeCorr.On.FitPar.egsigma=(aRun->getFitParErr())[3]; 
  
      //==========S/N===========//
      sdsDet[iDet]->StoN.On.FitPar.entries=aRun->doFit(theRun.number,"Summary_ClusterStoN_OnTrack_in",DetList[iDet]);
      sdsDet[iDet]->StoN.On.FitPar.width=(aRun->getFitPar())[0];
      sdsDet[iDet]->StoN.On.FitPar.mp=(aRun->getFitPar())[1];
      sdsDet[iDet]->StoN.On.FitPar.area=(aRun->getFitPar())[2];
      sdsDet[iDet]->StoN.On.FitPar.gsigma=(aRun->getFitPar())[3];   
      sdsDet[iDet]->StoN.On.FitPar.peak=aRun->pLanConv[0];
      sdsDet[iDet]->StoN.On.FitPar.FWHM=aRun->pLanConv[1];
      sdsDet[iDet]->StoN.On.FitPar.mp_peak=(sdsDet[iDet]->StoN.On.FitPar.mp - sdsDet[iDet]->StoN.On.FitPar.peak);
      sdsDet[iDet]->StoN.On.FitPar.chi2red=aRun->getFitChi()/aRun->getFitnDof(); 

      sdsDet[iDet]->StoN.On.FitPar.ewidth=(aRun->getFitParErr())[0];
      sdsDet[iDet]->StoN.On.FitPar.emp=(aRun->getFitParErr())[1];
      sdsDet[iDet]->StoN.On.FitPar.earea=(aRun->getFitParErr())[2];
      sdsDet[iDet]->StoN.On.FitPar.egsigma=(aRun->getFitParErr())[3];   

      sdsDet[iDet]->StoN.Off.HistoPar.entries=aRun->doHisto("Summary_ClusterStoN_OnTrack_in",DetList[iDet]);
      sdsDet[iDet]->StoN.Off.HistoPar.mean=(aRun->getHistoPar())[0];
      sdsDet[iDet]->StoN.Off.HistoPar.rms=(aRun->getHistoPar())[1];

      sdsDet[iDet]->StoNCorr.On.FitPar.entries=aRun->doFit(theRun.number,"Summary_ClusterStoNCorr_OnTrack_in",DetList[iDet]);
      sdsDet[iDet]->StoNCorr.On.FitPar.width=(aRun->getFitPar())[0];
      sdsDet[iDet]->StoNCorr.On.FitPar.mp=(aRun->getFitPar())[1];
      sdsDet[iDet]->StoNCorr.On.FitPar.area=(aRun->getFitPar())[2];
      sdsDet[iDet]->StoNCorr.On.FitPar.gsigma=(aRun->getFitPar())[3]; 
      sdsDet[iDet]->StoNCorr.On.FitPar.peak=aRun->pLanConv[0];
      sdsDet[iDet]->StoNCorr.On.FitPar.FWHM=aRun->pLanConv[1];
      sdsDet[iDet]->StoNCorr.On.FitPar.mp_peak=(sdsDet[iDet]->StoNCorr.On.FitPar.mp - sdsDet[iDet]->StoNCorr.On.FitPar.peak);
      sdsDet[iDet]->StoNCorr.On.FitPar.chi2red=aRun->getFitChi()/aRun->getFitnDof(); 
 
      sdsDet[iDet]->StoNCorr.On.FitPar.ewidth=(aRun->getFitParErr())[0];
      sdsDet[iDet]->StoNCorr.On.FitPar.emp=(aRun->getFitParErr())[1];
      sdsDet[iDet]->StoNCorr.On.FitPar.earea=(aRun->getFitParErr())[2];
      sdsDet[iDet]->StoNCorr.On.FitPar.egsigma=(aRun->getFitParErr())[3];      
    }
    tree->Fill();
      
  }
  f->Write();
  
  
}

