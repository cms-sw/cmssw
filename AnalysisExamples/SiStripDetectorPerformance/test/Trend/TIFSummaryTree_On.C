#include "TFile.h"
#include "TTree.h"
#include "TH1.h"
#include "TF1.h"
#include "TROOT.h"
#include "TCanvas.h"
#include "TPaveText.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

std::vector<std::string> runListString;

bool debug = true;
using namespace std;

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

      // Convolution integral of Landau and Gaussian by sum
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

   // Seaches for the location (x value) at the maximum of the 
   // Landau-Gaussian convolute and its full width at half-maximum.
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
// The signal function: a gaussian
{
   Double_t arg = 0;
   if (par[2]) arg = (x[0] - par[1])/par[2];

   Double_t noise = par[0]*TMath::Exp(-0.5*arg*arg);
   return noise;
}

//----------------------------------------//
class TIFRun{

 public: 
  TIFRun(Int_t,std::string);  
  ~TIFRun();

 //====== Histo ======//
  Stat_t doHisto(Char_t*,Char_t*,Char_t*,Char_t* );
  Double_t *getHistoPar() {return pPar;} 
  Stat_t doGlobalHisto(Char_t*, Char_t*);
  Stat_t doClusterHisto(Char_t*, Char_t*);
  //====== Fit ======//
  Stat_t doFit(Int_t, Char_t*, Char_t* , Char_t*, Char_t* );
  Stat_t doNoiseFit(Int_t, Char_t*, Char_t*, Char_t*, Char_t*);
  Double_t* getFitPar() {return pLanGausS;} 
  Double_t *getNoisePar() {return pGausS;}
  Double_t* getFitParErr() {return epLanGausS;}
  Double_t* getNoiseParErr() {return pGausS;}
  Double_t getFitChi() {return chi2GausS;}
  Int_t getFitnDof() {return nDofGausS;}
  Int_t getRun(){return fRun;}

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
};

TIFRun::TIFRun(Int_t r, std::string RootFileName) : fRun(r) {
  Char_t RunName[12];
  sprintf(RunName,"%d",r);
  fName=new TFile(RootFileName.c_str());
  if (! fName->IsOpen()){
    cout << "PROBLEM: File " << RootFileName << " is not open" << endl;
  }
  pPar=new Double_t[3];
  pLanGausS=new Double_t[4];
  pGausS=new Double_t[3];
  epGausS=new Double_t[3];
  epLanGausS=new Double_t[4];
  pLanConv=new Double_t[2];  
  langausFit=0;
  gausFit=0;
}

TIFRun::~TIFRun(){
  if ( fName!=0 ) delete  fName;
  if ( pPar!=0 ) delete [] pPar;
  if ( pLanGausS!=0 ) delete [] pLanGausS;
  if ( epLanGausS!=0 ) delete [] epLanGausS;
  if ( pGausS!=0 ) delete [] pGausS;
  if ( epGausS!=0 ) delete [] epGausS;
  if ( pLanConv!=0) delete[] pLanConv;
  if ( langausFit!=0 ) delete langausFit;
  if ( gausFit!=0 ) delete gausFit;

}

// Noise section
Stat_t TIFRun::doHisto(Char_t *FolderName, Char_t * Variable, Char_t *SubDetName, Char_t *TypeName){
  pPar[0]=0; pPar[1]=0; pPar[2]=0;
  Char_t hNameH[30];
  sprintf(hNameH,"%s/%s%s_%s",FolderName, Variable, SubDetName, TypeName);
  //cout<< hNameH << endl;  
  TH1F *hHisto=NULL;  
  fName->GetObject(hNameH,hHisto);
  if (hHisto==NULL){
    cout << "[doHisto] WARNING Histo " << hNameH << " doesn't exist\n\t\tReporting default value -10" << endl;
    return -15;
  }
  if(hHisto->GetEntries()!=0) {
    pPar[0]=hHisto->GetMean();
    pPar[1]=hHisto->GetRMS();  
  }
  else{
    cout<<"Empty Histogram "<< hNameH << endl;
    pPar[0]=-10; pPar[1]=-10;
  }
  return hHisto->GetEntries();
}

Stat_t TIFRun::doGlobalHisto(Char_t *FolderName, Char_t * Variable){
 Double_t Entries;
  pPar[0]=0; pPar[1]=0; pPar[2]=0;
  Char_t hNameH1[30];
  sprintf(hNameH1,"%s/%s",FolderName, Variable);
  TH1F *hHisto1=NULL;  
  fName->GetObject(hNameH1,hHisto1);
  if (hHisto1==NULL){
    cout << "[doGlobalHisto] WARNING Histo " << hNameH1 << " doesn't exist\n\t\tReporting default value -15" << endl;
    //    pPar[0]=-10; pPar[1]=-10;
    return -15;
  }
  Entries=hHisto1->GetEntries(); 
  if(Entries!=0) {
    pPar[0]=hHisto1->GetMean();
    pPar[1]=hHisto1->GetRMS();
    pPar[2]=hHisto1->GetBinContent(1);
  }
  else{
    cout<<"Empty Histogram "<< hNameH1 << endl;
    pPar[0]=-10; pPar[1]=-10;  pPar[2]=-10;
    }
  return Entries;  
}

Stat_t TIFRun::doClusterHisto(Char_t *FolderName, Char_t * Variable){
  Double_t Entries, Nzero;

  pPar[0]=0; pPar[1]=0;
  Char_t hNameH1[30];
  sprintf(hNameH1,"%s/%s",FolderName, Variable);
  TH1F *hHisto1=NULL;  
  fName->GetObject(hNameH1,hHisto1);
  if (hHisto1==NULL){
    cout << "[doClusterHisto] WARNING Histo " << hNameH1 << " doesn't exist\n\t\tReporting default value -15" << endl;
    //    pPar[0]=-10; pPar[1]=-10;
    return -15;
  }

  Entries=hHisto1->GetEntries();
  Nzero=hHisto1->GetBinContent(1); 
  cout << "Entries: "<< Entries << "N0: " << Nzero << endl;
 
  if(Entries!=0) {
    pPar[0]=hHisto1->GetMean();
    cout << "Mean: " << pPar[0] << endl;
    pPar[1]=(pPar[0]*Entries)/(Entries-Nzero); 
    cout << "Mean corrected: " << pPar[1] << endl;  
 
  }
  else{
    cout<<"Empty Histogram "<< hNameH1 << endl;
    pPar[0]=-10; pPar[1]=-10;
    }


  return Entries;  
}

// Signal section
Stat_t TIFRun::doFit(Int_t RunNumber, Char_t *FolderName, Char_t *Variable, Char_t *SubDetName, Char_t *TypeName){
  pLanGausS[0]=0; pLanGausS[1]=0; pLanGausS[2]=0; pLanGausS[3]=0;
  epLanGausS[0]=0; epLanGausS[1]=0; epLanGausS[2]=0; epLanGausS[3]=0;

  //FIXMEEEEE
  Char_t hNameF[32];
  sprintf(hNameF,"%s/%s%s_%s",FolderName, Variable, SubDetName,TypeName);
  TH1F *htoFit=NULL;
  fName->GetObject(hNameF,htoFit);
 if (htoFit==NULL){
   cout << "[doFit] WARNING Histo " << hNameF << " doesn't exist\n\t\tReporting default value -15" << endl;
    return -15;
  }
  
  if (htoFit->GetEntries()!=0) {
    cout<<"Fitting "<< hNameF <<endl;
    // Setting fit range and start values
    Double_t fr[2];
    Double_t sv[4], pllo[4], plhi[4];
    fr[0]=0.45*htoFit->GetMean();
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
    
    chi2GausS =langausFit->GetChisquare();  // obtain chi^2
    nDofGausS = langausFit->GetNDF();           // obtain ndf
    
    Double_t sPeak, sFWHM;
    langaupro(pLanGausS,sPeak,sFWHM);
    pLanConv[0]=sPeak;
    pLanConv[1]=sFWHM;
    cout << "langaupro:  max  " << sPeak << endl;
    cout << "langaupro:  FWHM " << sFWHM << endl;

    TCanvas *cAll = new TCanvas("Fit",hNameF,1);
    Char_t fitFileName[60];
    sprintf(fitFileName,"Fits/Fit_%d_%s_%s_%s_new.root",RunNumber, Variable, SubDetName,TypeName);
    htoFit->Draw("pe");
    htoFit->SetStats(100);
    langausFit->Draw("lsame");

    cAll->Print(fitFileName,"root");
  }
  else {  
    cout << "[doFit] WARNING : Histo " << hNameF << " is empty. Setting default values" << endl;
    pLanGausS[0]=-10; pLanGausS[1]=-10; pLanGausS[2]=-10; pLanGausS[3]=-10;
    epLanGausS[0]=-10; epLanGausS[1]=-10; epLanGausS[2]=-10; epLanGausS[3]=-10;
    pLanConv[0]=-10;   pLanConv[1]=-10;   chi2GausS=-10;  nDofGausS=-10;    
  }

  return htoFit->GetEntries();

}

//Noise sction
Stat_t TIFRun::doNoiseFit(Int_t RunNumber, Char_t *FolderName, Char_t *Variable, Char_t *SubDetName, Char_t *TypeName){
  Char_t hNameF[32];
  sprintf(hNameF,"%s/%s%s_%s",FolderName, Variable, SubDetName,TypeName);
  TH1F* hNtoFit=NULL;
  fName->GetObject(hNameF,hNtoFit);
  if (hNtoFit==NULL){
    cout << "[doNoiseFit] WARNING Histo " << hNameF << " doesn't exist\n\t\tReporting default value -15" << endl;
    return -15;
  }
  
  if (hNtoFit->GetEntries()!=0) {
    cout<<"Fitting "<< hNameF <<endl;
    // Setting fit range and start values
    Double_t fr[2];
    Double_t sv[3], pllo[3], plhi[3];
    fr[0]=0.3*hNtoFit->GetMean();
    fr[1]=3.0*hNtoFit->GetMean();

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
    
    //    chi2GausS =langausFit->GetChisquare();  // obtain chi^2
    //    nDofGausS = langausFit->GetNDF();           // obtain ndf
 
//     TCanvas *cAllN = new TCanvas("NoiseFit",hNameF,1);
//     Char_t fitFileName[60];
//     sprintf(fitFileName,"~/backup/Fits/FitNoise_%d_%s_%s_%s.root",RunNumber, Variable, SubDetName,TypeName);
//     hNtoFit->Draw("pe");
//     hNtoFit->SetStats(1111110);
//     gausFit->Draw("lsame");

//     cAllN->Print(fitFileName,"root");
  }
  else {  
    pGausS[0]=-10; pGausS[1]=-10; pGausS[2]=-10;
    epGausS[0]=-10; epGausS[1]=-10; epGausS[2]=-10;
  }

  return hNtoFit->GetEntries();
    
}


//---------- Summary Tree --------------// 

struct RunInfo_t {
  Int_t number;
  // maybe add here latency, APV mode...
};

struct ClusterValues_t{
  Stat_t entries;
  Stat_t entries_all;
  Double_t mean; 
  Double_t mean_corr;

};

struct HistoValues_t { 
  Stat_t entries;
  Double_t mean; 
  Double_t rms;
  Double_t N0;
  Double_t MeanTrack;
}; 

struct LanGausValues_t { 
  Stat_t entries;
  Double_t width, ewidth;
  Double_t mp, emp;
  Double_t area, earea;
  Double_t gsigma,  egsigma;
  Double_t peak, FWHM;
  Double_t mp_peak;
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
  SubDetSummary_t All;
  SubDetSummary_t On;
  SubDetSummary_t Off;
};

struct LayerSummary_t {
  SubDetSummary_t L1;
  SubDetSummary_t L2;
  SubDetSummary_t L3;
  SubDetSummary_t L4;
  SubDetSummary_t L5;
  SubDetSummary_t L6;
  SubDetSummary_t L7;
  SubDetSummary_t L8;
  SubDetSummary_t L9;

};
struct FinalSummary_t {
  TotSummary_t SignalCorr;
  TotSummary_t Noise;
  TotSummary_t StoNCorr;
  TotSummary_t Width;
  TotSummary_t nClusters;
  LayerSummary_t cSignalCorr;
  LayerSummary_t cSignal;
  LayerSummary_t cStoN;
  LayerSummary_t cWidth;
  LayerSummary_t cNoise;
  LayerSummary_t cStoNCorr;
};

//-----------------------------------------------//

//========== Per laayer===========//
void SignalCorrLayer(Int_t iDet,int runList,TIFRun* aRun, FinalSummary_t* sdsDet,char* DetList);
void StoNCorrLayer(Int_t iDet,int runList,TIFRun* aRun, FinalSummary_t* sdsDet,char* DetList);
void NoiseAndWidthLayer(Int_t iDet,int runList,TIFRun* aRun, FinalSummary_t* sdsDet,char* DetList);

//========== Per SubDet =========//
void SignalCorr(int runList,TIFRun* aRun, FinalSummary_t* sdsDet,char* DetList);
void StoNCorr(int runList,TIFRun* aRun, FinalSummary_t* sdsDet,char* DetList);
void NoiseAndWidth(TIFRun* aRun,int runList,FinalSummary_t* sdsDet,char* DetList);

void TIFSummaryTree(char* cRunList="", char* outputFile="") {  

  // loop on the runs
  std::vector<int> runList;
  
  ifstream infile;
  infile.open(cRunList);
  int irun;
  char srun[2048];
  while (infile.good()){
    infile >> irun >> srun;
    runList.push_back(irun);
    runListString.push_back(srun);
    cout << irun << " " << srun << endl;
  }
  infile.close();

  const Int_t nRun=runList.size();

  if (nRun==0){
    std::cout<< "PROBLEM: input file " << cRunList << " is empty\n EXIT" << std::endl;
    return;
  }


  // tree leaves
  RunInfo_t theRun;
  HistoValues_t *theTracks;
  HistoValues_t *theRecHits;
  ClusterValues_t *theClusters;
  FinalSummary_t* sdsDet[4];

  for(Int_t iDet=0; iDet<4;iDet++) {
    sdsDet[iDet]=new FinalSummary_t;
  }

  theTracks = new HistoValues_t;
  theRecHits = new HistoValues_t;
  theClusters = new ClusterValues_t;

  if(debug)
    cout<<"TTree to be initialized"<<endl;
  // tree declarations
  TFile *f=new TFile(outputFile,"RECREATE");
  TTree *tree=new TTree("TIFTree","Summary Tree for TIF data");
  tree->Branch("Run",&theRun,"number/I");
  tree->Branch("Tracks.","HistoValues_t", &theTracks, 32000, 2);
  tree->Branch("RecHits.","HistoValues_t",&theRecHits, 32000, 2);
  tree->Branch("Clusters.","ClusterValues_t",&theClusters, 32000, 2);

  tree->Branch("TIB.", "FinalSummary_t",&(sdsDet[0]), 32000, 2);
  tree->Branch("TOB.", "FinalSummary_t",&(sdsDet[1]), 32000, 2);
  tree->Branch("TEC.", "FinalSummary_t",&(sdsDet[2]), 32000, 2);
  tree->Branch("TID.", "FinalSummary_t",&(sdsDet[3]), 32000, 2);
  if(debug)
    cout<<"TTree initialized"<<endl;


  Char_t *DetList[4] = {"TIB","TOB","TEC","TID"};
  TIFRun *aRun;

  cout << "... Number of runs in RunList " << nRun<< endl;
  for (Int_t iRun=0; iRun<nRun; iRun++){    
    cout << "... Looking at file " << runListString[iRun] << endl;
    aRun = new TIFRun(runList[iRun],runListString[iRun]);
    theRun.number=aRun->getRun();
    cout << "=================> RUN NUMBER: " << theRun.number <<endl;

    //============Global Histos!!=======////

    theClusters->entries=aRun->doClusterHisto("Tracks","nClusters_onTrack");
    theClusters->mean=(aRun->getHistoPar())[0];
    theClusters->mean_corr=(aRun->getHistoPar())[1];
    theClusters->entries_all=aRun->doClusterHisto("Tracks","nClusters_All");//gives the total event number

    theTracks->entries=aRun->doGlobalHisto("Tracks","nTracks");
    theTracks->mean=(aRun->getHistoPar())[0];
    theTracks->rms=(aRun->getHistoPar())[1];
    theTracks->N0=(aRun->getHistoPar())[2];
    theTracks->MeanTrack=((theTracks->entries-theTracks->N0)/theClusters->entries_all);

    cout << "Tracks end" << endl;
    
    theRecHits->entries=aRun->doGlobalHisto("Tracks","nRecHits");
    theRecHits->mean=(aRun->getHistoPar())[0];
    theRecHits->rms=(aRun->getHistoPar())[1];

    cout << "Total number of events from Clusters_All" << theClusters->entries_all << endl;

  cout << "MeanTrack: "<<  theTracks->MeanTrack << endl;

  //    cout << "Clusters mean corr: " <<  theClusters->mean_corr << endl;

    for(Int_t iDet=0; iDet<4;iDet++) {
      if(debug)
	cout<<"LOOP on Det"<<endl;

      SignalCorrLayer(iDet,runList[iRun],aRun,sdsDet[iDet],DetList[iDet]);
      StoNCorrLayer(iDet,runList[iRun],aRun,sdsDet[iDet],DetList[iDet]);
      NoiseAndWidthLayer(iDet,runList[iRun],aRun,sdsDet[iDet],DetList[iDet]);

      NoiseAndWidth(aRun,runList[iRun],sdsDet[iDet],DetList[iDet]);
      SignalCorr(runList[iRun],aRun,sdsDet[iDet],DetList[iDet]);
      StoNCorr(runList[iRun],aRun,sdsDet[iDet],DetList[iDet]);
    }
  
  tree->Fill();   
  }
  f->Write();
}

void NoiseAndWidth(TIFRun* aRun,int runList,FinalSummary_t* sdsDet,char* DetList){
  cout << "...ClusterNoise Called... "<< endl;
  sdsDet->Noise.On.HistoPar.entries=aRun->doHisto("ClusterNoise","cNoise_",DetList,"onTrack");
  sdsDet->Noise.On.HistoPar.mean=(aRun->getHistoPar())[0];
  sdsDet->Noise.On.HistoPar.rms=(aRun->getHistoPar())[1];
  
  sdsDet->Noise.On.FitNoisePar.entries=aRun->doNoiseFit(runList,"ClusterNoise","cNoise_",DetList,"onTrack");
  sdsDet->Noise.On.FitNoisePar.garea=(aRun->getNoisePar())[0]; 
  sdsDet->Noise.On.FitNoisePar.fitmean=(aRun->getNoisePar())[1];
  sdsDet->Noise.On.FitNoisePar.fitrms=(aRun->getNoisePar())[2];
  
  sdsDet->Noise.On.FitNoisePar.egarea=(aRun->getNoiseParErr())[0];
  sdsDet->Noise.On.FitNoisePar.efitmean=(aRun->getNoiseParErr())[1];
  sdsDet->Noise.On.FitNoisePar.efitrms=(aRun->getNoiseParErr())[2];

  sdsDet->Width.On.HistoPar.entries=aRun->doHisto("ClusterWidth","cWidth_",DetList,"onTrack");
  sdsDet->Width.On.HistoPar.mean=(aRun->getHistoPar())[0];
  sdsDet->Width.On.HistoPar.rms=(aRun->getHistoPar())[1];
  }
void SignalCorr(int runList,TIFRun* aRun, FinalSummary_t* sdsDet,char* DetList){

      sdsDet->SignalCorr.On.HistoPar.entries=aRun->doHisto("ClusterSignal","cSignalCorr_",DetList,"onTrack");
      sdsDet->SignalCorr.On.HistoPar.mean=(aRun->getHistoPar())[0];
      sdsDet->SignalCorr.On.HistoPar.rms=(aRun->getHistoPar())[1];
      


      sdsDet->SignalCorr.On.FitPar.entries=aRun->doFit(runList,"ClusterSignal","cSignalCorr_",DetList,"onTrack");
      sdsDet->SignalCorr.On.FitPar.mp=(aRun->getFitPar())[1];
      sdsDet->SignalCorr.On.FitPar.width=(aRun->getFitPar())[0];
      sdsDet->SignalCorr.On.FitPar.area=(aRun->getFitPar())[2];
      sdsDet->SignalCorr.On.FitPar.gsigma=(aRun->getFitPar())[3];
      sdsDet->SignalCorr.On.FitPar.mp_peak=(sdsDet->SignalCorr.On.FitPar.mp - sdsDet->SignalCorr.On.FitPar.peak);

      sdsDet->SignalCorr.On.FitPar.peak=aRun->pLanConv[0];
      sdsDet->SignalCorr.On.FitPar.FWHM=aRun->pLanConv[1];

      sdsDet->SignalCorr.On.FitPar.ewidth=(aRun->getFitParErr())[0];
      sdsDet->SignalCorr.On.FitPar.emp=(aRun->getFitParErr())[1];
      sdsDet->SignalCorr.On.FitPar.earea=(aRun->getFitParErr())[2];
      sdsDet->SignalCorr.On.FitPar.egsigma=(aRun->getFitParErr())[3];

}

void StoNCorr(int runList,TIFRun* aRun, FinalSummary_t* sdsDet,char* DetList) {
      sdsDet->StoNCorr.On.HistoPar.entries=aRun->doHisto("ClusterStoN","cStoNCorr_",DetList,"onTrack");
      sdsDet->StoNCorr.On.HistoPar.mean=(aRun->getHistoPar())[0];
      sdsDet->StoNCorr.On.HistoPar.rms=(aRun->getHistoPar())[1];

      sdsDet->StoNCorr.On.FitPar.entries=aRun->doFit(runList,"ClusterStoN","cStoNCorr_",DetList,"onTrack");
      sdsDet->StoNCorr.On.FitPar.mp=(aRun->getFitPar())[1];
      sdsDet->StoNCorr.On.FitPar.width=(aRun->getFitPar())[0];
      sdsDet->StoNCorr.On.FitPar.area=(aRun->getFitPar())[2];
      sdsDet->StoNCorr.On.FitPar.gsigma=(aRun->getFitPar())[3];
      sdsDet->StoNCorr.On.FitPar.mp_peak=(sdsDet->StoNCorr.On.FitPar.mp - sdsDet->StoNCorr.On.FitPar.peak);
      
      sdsDet->StoNCorr.On.FitPar.peak=aRun->pLanConv[0];
      sdsDet->StoNCorr.On.FitPar.FWHM=aRun->pLanConv[1];
      
      sdsDet->StoNCorr.On.FitPar.ewidth=(aRun->getFitParErr())[0];
      sdsDet->StoNCorr.On.FitPar.emp=(aRun->getFitParErr())[1];
      sdsDet->StoNCorr.On.FitPar.earea=(aRun->getFitParErr())[2];
      sdsDet->StoNCorr.On.FitPar.egsigma=(aRun->getFitParErr())[3];
}

void NoiseAndWidthLayer(Int_t iDet,int runList,TIFRun* aRun, FinalSummary_t* sdsDet,char* DetList){		  

  cout << "...NoiseLayer Called... "<<endl;															     
  sdsDet->cNoise.L1.HistoPar.entries=aRun->doHisto("Layer","cNoise_",DetList,"Layer_1_onTrack");					     
  sdsDet->cNoise.L1.HistoPar.mean=(aRun->getHistoPar())[0];													     
  sdsDet->cNoise.L1.HistoPar.rms=(aRun->getHistoPar())[1];													     
  
  sdsDet->cNoise.L1.FitNoisePar.entries=aRun->doNoiseFit(runList,"Layer","cNoise_",DetList,"Layer_1_onTrack");							     
  sdsDet->cNoise.L1.FitNoisePar.garea=(aRun->getNoisePar())[0];												     
  sdsDet->cNoise.L1.FitNoisePar.fitmean=(aRun->getNoisePar())[1];												     
  sdsDet->cNoise.L1.FitNoisePar.fitrms=(aRun->getNoisePar())[2];												     
  
  sdsDet->cNoise.L1.FitNoisePar.egarea=(aRun->getNoiseParErr())[0];												     
  sdsDet->cNoise.L1.FitNoisePar.efitmean=(aRun->getNoiseParErr())[1];												     
  sdsDet->cNoise.L1.FitNoisePar.efitrms=(aRun->getNoiseParErr())[2];												     
  
  sdsDet->cNoise.L2.HistoPar.entries=aRun->doHisto("Layer","cNoise_",DetList,"Layer_2_onTrack");								     
  sdsDet->cNoise.L2.HistoPar.mean=(aRun->getHistoPar())[0];													     
  sdsDet->cNoise.L2.HistoPar.rms=(aRun->getHistoPar())[1];													     
  
  sdsDet->cNoise.L2.FitNoisePar.entries=aRun->doNoiseFit(runList,"Layer","cNoise_",DetList,"Layer_2_onTrack");							     
  sdsDet->cNoise.L2.FitNoisePar.garea=(aRun->getNoisePar())[0];												     
  sdsDet->cNoise.L2.FitNoisePar.fitmean=(aRun->getNoisePar())[1];												     
  sdsDet->cNoise.L2.FitNoisePar.fitrms=(aRun->getNoisePar())[2];												     
  
  sdsDet->cNoise.L2.FitNoisePar.egarea=(aRun->getNoiseParErr())[0];												     
  sdsDet->cNoise.L2.FitNoisePar.efitmean=(aRun->getNoiseParErr())[1];												     
  sdsDet->cNoise.L2.FitNoisePar.efitrms=(aRun->getNoiseParErr())[2];												     
  
  sdsDet->cNoise.L3.HistoPar.entries=aRun->doHisto("Layer","cNoise_",DetList,"Layer_3_onTrack");								     
  sdsDet->cNoise.L3.HistoPar.mean=(aRun->getHistoPar())[0];													     
  sdsDet->cNoise.L3.HistoPar.rms=(aRun->getHistoPar())[1];													     
  
  sdsDet->cNoise.L3.FitNoisePar.entries=aRun->doNoiseFit(runList,"Layer","cNoise_",DetList,"Layer_3_onTrack");							     
  
  sdsDet->cNoise.L3.FitNoisePar.garea=(aRun->getNoisePar())[0];												     
  sdsDet->cNoise.L3.FitNoisePar.fitmean=(aRun->getNoisePar())[1];												     
  sdsDet->cNoise.L3.FitNoisePar.fitrms=(aRun->getNoisePar())[2];												     
  
  sdsDet->cNoise.L3.FitNoisePar.egarea=(aRun->getNoiseParErr())[0];												     
  sdsDet->cNoise.L3.FitNoisePar.efitmean=(aRun->getNoiseParErr())[1];												     
  sdsDet->cNoise.L3.FitNoisePar.efitrms=(aRun->getNoiseParErr())[2];												     
  
   //   cout <<"=========Width================" << endl;													     
   sdsDet->cWidth.L1.HistoPar.entries=aRun->doHisto("Layer","cWidth_",DetList,"Layer_1_onTrack");								     
   sdsDet->cWidth.L1.HistoPar.mean=(aRun->getHistoPar())[0];													     
   sdsDet->cWidth.L1.HistoPar.rms=(aRun->getHistoPar())[1];													     
   																				     
   sdsDet->cWidth.L2.HistoPar.entries=aRun->doHisto("Layer","cWidth_",DetList,"Layer_2_onTrack");								     
   sdsDet->cWidth.L2.HistoPar.mean=(aRun->getHistoPar())[0];													     
   sdsDet->cWidth.L2.HistoPar.rms=(aRun->getHistoPar())[1];													     
   																				     
   sdsDet->cWidth.L3.HistoPar.entries=aRun->doHisto("Layer","cWidth_",DetList,"Layer_3_onTrack");								     
   sdsDet->cWidth.L3.HistoPar.mean=(aRun->getHistoPar())[0];													     
   sdsDet->cWidth.L3.HistoPar.rms=(aRun->getHistoPar())[1];													     
																				     
   if(iDet<3){																			     
     sdsDet->cNoise.L4.HistoPar.entries=aRun->doHisto("Layer","cNoise_",DetList,"Layer_4_onTrack");								     
     sdsDet->cNoise.L4.HistoPar.mean=(aRun->getHistoPar())[0];													     
     sdsDet->cNoise.L4.HistoPar.rms=(aRun->getHistoPar())[1];													     
     																				     
     sdsDet->cNoise.L4.FitNoisePar.entries=aRun->doNoiseFit(runList,"Layer","cNoise_",DetList,"Layer_4_onTrack");						     
     sdsDet->cNoise.L4.FitNoisePar.garea=(aRun->getNoiseParErr())[0];												     
     sdsDet->cNoise.L4.FitNoisePar.fitmean=(aRun->getNoisePar())[1];												     
     sdsDet->cNoise.L4.FitNoisePar.fitrms=(aRun->getNoisePar())[2];												     
     																				     
     sdsDet->cNoise.L4.FitNoisePar.egarea=(aRun->getNoiseParErr())[0];												     
     sdsDet->cNoise.L4.FitNoisePar.efitmean=(aRun->getNoiseParErr())[1];											     
     sdsDet->cNoise.L4.FitNoisePar.efitrms=(aRun->getNoiseParErr())[2];												     
     																				     
     sdsDet->cWidth.L4.HistoPar.entries=aRun->doHisto("Layer","cWidth_",DetList,"Layer_4_onTrack");								     
     sdsDet->cWidth.L4.HistoPar.mean=(aRun->getHistoPar())[0];													     
     sdsDet->cWidth.L4.HistoPar.rms=(aRun->getHistoPar())[1];													     
     																				     
   }																				     
   if(iDet==1||iDet==2) { 																	     
     sdsDet->cNoise.L5.HistoPar.entries=aRun->doHisto("Layer","cNoise_",DetList,"Layer_5_onTrack");								     
     sdsDet->cNoise.L5.HistoPar.mean=(aRun->getHistoPar())[0];													     
     sdsDet->cNoise.L5.HistoPar.rms=(aRun->getHistoPar())[1];													     
     																				     
     sdsDet->cNoise.L5.FitNoisePar.entries=aRun->doNoiseFit(runList,"Layer","cNoise_",DetList,"Layer_5_onTrack");						     
     sdsDet->cNoise.L5.FitNoisePar.garea=(aRun->getNoiseParErr())[0];												     
     sdsDet->cNoise.L5.FitNoisePar.fitmean=(aRun->getNoisePar())[1];												     
     sdsDet->cNoise.L5.FitNoisePar.fitrms=(aRun->getNoisePar())[2];												     
     																				     
     sdsDet->cNoise.L5.FitNoisePar.egarea=(aRun->getNoiseParErr())[0];												     
     sdsDet->cNoise.L5.FitNoisePar.efitmean=(aRun->getNoiseParErr())[1];											     
     sdsDet->cNoise.L5.FitNoisePar.efitrms=(aRun->getNoiseParErr())[2];												     
     																				     
     sdsDet->cNoise.L6.HistoPar.entries=aRun->doHisto("Layer","cNoise_",DetList,"Layer_6_onTrack");								     
     sdsDet->cNoise.L6.HistoPar.mean=(aRun->getHistoPar())[0];													     
     sdsDet->cNoise.L6.HistoPar.rms=(aRun->getHistoPar())[1];													     
     																				     
     sdsDet->cNoise.L6.FitNoisePar.entries=aRun->doNoiseFit(runList,"Layer","cNoise_",DetList,"Layer_6_onTrack");						     
     sdsDet->cNoise.L6.FitNoisePar.garea=(aRun->getNoiseParErr())[0];												     
     sdsDet->cNoise.L6.FitNoisePar.fitmean=(aRun->getNoisePar())[1];												     
     sdsDet->cNoise.L6.FitNoisePar.fitrms=(aRun->getNoisePar())[2];												     
     																				     
     sdsDet->cNoise.L6.FitNoisePar.egarea=(aRun->getNoiseParErr())[0];												     
     sdsDet->cNoise.L6.FitNoisePar.efitmean=(aRun->getNoiseParErr())[1];											     
     sdsDet->cNoise.L6.FitNoisePar.efitrms=(aRun->getNoiseParErr())[2];												     
     																				     
     sdsDet->cWidth.L5.HistoPar.entries=aRun->doHisto("Layer","cWidth_",DetList,"Layer_5_onTrack");								     
     sdsDet->cWidth.L5.HistoPar.mean=(aRun->getHistoPar())[0];													     
     sdsDet->cWidth.L5.HistoPar.rms=(aRun->getHistoPar())[1];													     
     																				     
     sdsDet->cWidth.L6.HistoPar.entries=aRun->doHisto("Layer","cWidth_",DetList,"Layer_6_onTrack");								     
     sdsDet->cWidth.L6.HistoPar.mean=(aRun->getHistoPar())[0];													     
     sdsDet->cWidth.L6.HistoPar.rms=(aRun->getHistoPar())[1];													     
   }																				     
   																				     
   if(iDet==2){																			     
     sdsDet->cNoise.L7.HistoPar.entries=aRun->doHisto("Layer","cNoise_",DetList,"Layer_7_onTrack");								     
     sdsDet->cNoise.L7.HistoPar.mean=(aRun->getHistoPar())[0];													     
     sdsDet->cNoise.L7.HistoPar.rms=(aRun->getHistoPar())[1];													     
     																				     
     sdsDet->cNoise.L7.FitNoisePar.entries=aRun->doNoiseFit(runList,"Layer","cNoise_",DetList,"Layer_7_onTrack");						     
     sdsDet->cNoise.L7.FitNoisePar.garea=(aRun->getNoiseParErr())[0];												     
     sdsDet->cNoise.L7.FitNoisePar.fitmean=(aRun->getNoisePar())[1];												     
     sdsDet->cNoise.L7.FitNoisePar.fitrms=(aRun->getNoisePar())[2];												     
     																				     
     sdsDet->cNoise.L7.FitNoisePar.egarea=(aRun->getNoiseParErr())[0];												     
     sdsDet->cNoise.L7.FitNoisePar.efitmean=(aRun->getNoiseParErr())[1];											     
     sdsDet->cNoise.L7.FitNoisePar.efitrms=(aRun->getNoiseParErr())[2];												     
     																				     
     sdsDet->cNoise.L8.HistoPar.entries=aRun->doHisto("Layer","cNoise_",DetList,"Layer_8_onTrack");								     
     sdsDet->cNoise.L8.HistoPar.mean=(aRun->getHistoPar())[0];													     
     sdsDet->cNoise.L8.HistoPar.rms=(aRun->getHistoPar())[1];													     
     																				     
     sdsDet->cNoise.L8.FitNoisePar.entries=aRun->doNoiseFit(runList,"Layer","cNoise_",DetList,"Layer_8_onTrack");						     
     sdsDet->cNoise.L8.FitNoisePar.garea=(aRun->getNoiseParErr())[0];												     
     sdsDet->cNoise.L8.FitNoisePar.fitmean=(aRun->getNoisePar())[1];												     
     sdsDet->cNoise.L8.FitNoisePar.fitrms=(aRun->getNoisePar())[2];												     
     																				     
     sdsDet->cNoise.L8.FitNoisePar.egarea=(aRun->getNoiseParErr())[0];												     
     sdsDet->cNoise.L8.FitNoisePar.efitmean=(aRun->getNoiseParErr())[1];											     
     sdsDet->cNoise.L8.FitNoisePar.efitrms=(aRun->getNoiseParErr())[2];												     
     																				     
     sdsDet->cNoise.L9.HistoPar.entries=aRun->doHisto("Layer","cNoise_",DetList,"Layer_9_onTrack");								     
     sdsDet->cNoise.L9.HistoPar.mean=(aRun->getHistoPar())[0];													     
     sdsDet->cNoise.L9.HistoPar.rms=(aRun->getHistoPar())[1];													     
     																				     
     sdsDet->cNoise.L9.FitNoisePar.entries=aRun->doNoiseFit(runList,"Layer","cNoise_",DetList,"Layer_9_onTrack");						     
     sdsDet->cNoise.L9.FitNoisePar.garea=(aRun->getNoiseParErr())[0];												     
     sdsDet->cNoise.L9.FitNoisePar.fitmean=(aRun->getNoisePar())[1];												     
     sdsDet->cNoise.L9.FitNoisePar.fitrms=(aRun->getNoisePar())[2];												     
     																				     
     sdsDet->cNoise.L9.FitNoisePar.egarea=(aRun->getNoiseParErr())[0];												     
     sdsDet->cNoise.L9.FitNoisePar.efitmean=(aRun->getNoiseParErr())[1];											     
     sdsDet->cNoise.L9.FitNoisePar.efitrms=(aRun->getNoiseParErr())[2];												     
     																				     
     sdsDet->cWidth.L7.HistoPar.entries=aRun->doHisto("Layer","cWidth_",DetList,"Layer_7_onTrack");								     
     sdsDet->cWidth.L7.HistoPar.mean=(aRun->getHistoPar())[0];													     
     sdsDet->cWidth.L7.HistoPar.rms=(aRun->getHistoPar())[1];													     
     																				     
     sdsDet->cWidth.L8.HistoPar.entries=aRun->doHisto("Layer","cWidth_",DetList,"Layer_8_onTrack");								     
     sdsDet->cWidth.L8.HistoPar.mean=(aRun->getHistoPar())[0];													     
     sdsDet->cWidth.L8.HistoPar.rms=(aRun->getHistoPar())[1];													     
     																				     
     sdsDet->cWidth.L9.HistoPar.entries=aRun->doHisto("Layer","cWidth_",DetList,"Layer_9_onTrack");								     
     sdsDet->cWidth.L9.HistoPar.mean=(aRun->getHistoPar())[0];													     
     sdsDet->cWidth.L9.HistoPar.rms=(aRun->getHistoPar())[1];													     
   }																				     
																				     
 }     																				     
																				     
   void SignalCorrLayer(Int_t iDet,int runList, TIFRun* aRun,FinalSummary_t* sdsDet,char* DetList){ 								     
     cout << "...SignalCorr Called... "<< endl; 														     
     sdsDet->cSignalCorr.L1.HistoPar.entries=aRun->doHisto("Layer","cSignalCorr_",DetList,"Layer_1_onTrack");							     
     sdsDet->cSignalCorr.L1.HistoPar.mean=(aRun->getHistoPar())[0];												     
     sdsDet->cSignalCorr.L1.HistoPar.rms=(aRun->getHistoPar())[1];												     
     																				     
     sdsDet->cSignalCorr.L2.HistoPar.entries=aRun->doHisto("Layer","cSignalCorr_",DetList,"Layer_2_onTrack");							     
     sdsDet->cSignalCorr.L2.HistoPar.mean=(aRun->getHistoPar())[0];												     
     sdsDet->cSignalCorr.L2.HistoPar.rms=(aRun->getHistoPar())[1];												     
     																				     
     sdsDet->cSignalCorr.L3.HistoPar.entries=aRun->doHisto("Layer","cSignalCorr_",DetList,"Layer_3_onTrack");							     
     sdsDet->cSignalCorr.L3.HistoPar.mean=(aRun->getHistoPar())[0];												     
     sdsDet->cSignalCorr.L3.HistoPar.rms=(aRun->getHistoPar())[1];												     
     																				     
     sdsDet->cSignalCorr.L1.FitPar.entries=aRun->doFit(runList,"Layer","cSignalCorr_",DetList,"Layer_1_onTrack");						     
     sdsDet->cSignalCorr.L1.FitPar.width=(aRun->getFitPar())[0];												     
     sdsDet->cSignalCorr.L1.FitPar.mp=(aRun->getFitPar())[1];													     
     sdsDet->cSignalCorr.L1.FitPar.area=(aRun->getFitPar())[2];													     
     sdsDet->cSignalCorr.L1.FitPar.gsigma=(aRun->getFitPar())[3];												     
     sdsDet->cSignalCorr.L1.FitPar.mp_peak=(sdsDet->cSignalCorr.L1.FitPar.mp - sdsDet->cSignalCorr.L1.FitPar.peak);						     
     																				     
     sdsDet->cSignalCorr.L1.FitPar.peak=aRun->pLanConv[0];													     
     sdsDet->cSignalCorr.L1.FitPar.FWHM=aRun->pLanConv[1];													     
     																				     
     sdsDet->cSignalCorr.L1.FitPar.ewidth=(aRun->getFitParErr())[0];												     
     sdsDet->cSignalCorr.L1.FitPar.emp=(aRun->getFitParErr())[1];												     
     sdsDet->cSignalCorr.L1.FitPar.earea=(aRun->getFitParErr())[2];												     
     sdsDet->cSignalCorr.L1.FitPar.egsigma=(aRun->getFitParErr())[3];												     
     																				     
     sdsDet->cSignalCorr.L2.FitPar.entries=aRun->doFit(runList,"Layer","cSignalCorr_",DetList,"Layer_2_onTrack");						     
     sdsDet->cSignalCorr.L2.FitPar.width=(aRun->getFitPar())[0];												     
     sdsDet->cSignalCorr.L2.FitPar.mp=(aRun->getFitPar())[1];													     
     sdsDet->cSignalCorr.L2.FitPar.area=(aRun->getFitPar())[2];													     
     sdsDet->cSignalCorr.L2.FitPar.gsigma=(aRun->getFitPar())[3];												     
     sdsDet->cSignalCorr.L2.FitPar.mp_peak=(sdsDet->cSignalCorr.L2.FitPar.mp - sdsDet->cSignalCorr.L2.FitPar.peak);						     
     																				     
     sdsDet->cSignalCorr.L2.FitPar.peak=aRun->pLanConv[0];													     
     sdsDet->cSignalCorr.L2.FitPar.FWHM=aRun->pLanConv[1];													     
     																				     
     sdsDet->cSignalCorr.L2.FitPar.ewidth=(aRun->getFitParErr())[0];												     
     sdsDet->cSignalCorr.L2.FitPar.emp=(aRun->getFitParErr())[1];												     
     sdsDet->cSignalCorr.L2.FitPar.earea=(aRun->getFitParErr())[2];												     
     sdsDet->cSignalCorr.L2.FitPar.egsigma=(aRun->getFitParErr())[3];												     
     																				     
     sdsDet->cSignalCorr.L3.FitPar.entries=aRun->doFit(runList,"Layer","cSignalCorr_",DetList,"Layer_3_onTrack");						     
     sdsDet->cSignalCorr.L3.FitPar.width=(aRun->getFitPar())[0];												     
     sdsDet->cSignalCorr.L3.FitPar.mp=(aRun->getFitPar())[1];													     
     sdsDet->cSignalCorr.L3.FitPar.area=(aRun->getFitPar())[2];													     
     sdsDet->cSignalCorr.L3.FitPar.gsigma=(aRun->getFitPar())[3];												     
     sdsDet->cSignalCorr.L3.FitPar.mp_peak=(sdsDet->cSignalCorr.L3.FitPar.mp - sdsDet->cSignalCorr.L3.FitPar.peak);						     
     																				     
     sdsDet->cSignalCorr.L3.FitPar.peak=aRun->pLanConv[0];													     
     sdsDet->cSignalCorr.L3.FitPar.FWHM=aRun->pLanConv[1];													     
     																				     
     sdsDet->cSignalCorr.L3.FitPar.ewidth=(aRun->getFitParErr())[0];												     
     sdsDet->cSignalCorr.L3.FitPar.emp=(aRun->getFitParErr())[1];												     
     sdsDet->cSignalCorr.L3.FitPar.earea=(aRun->getFitParErr())[2];												     
     sdsDet->cSignalCorr.L3.FitPar.egsigma=(aRun->getFitParErr())[3];												     
     																				     
     if (iDet !=3) {																		     
       																				     
       //cout << "================HISTO=================" << endl;												     
       sdsDet->cSignalCorr.L4.HistoPar.entries=aRun->doHisto("Layer","cSignalCorr_",DetList,"Layer_4_onTrack");							     
       sdsDet->cSignalCorr.L4.HistoPar.mean=(aRun->getHistoPar())[0];												     
       sdsDet->cSignalCorr.L4.HistoPar.rms=(aRun->getHistoPar())[1];												     
       //cout << "===============FIT=============" << endl;													     
       sdsDet->cSignalCorr.L4.FitPar.entries=aRun->doFit(runList,"Layer","cSignalCorr_",DetList,"Layer_4_onTrack");						     
       sdsDet->cSignalCorr.L4.FitPar.width=(aRun->getFitPar())[0];												     
       sdsDet->cSignalCorr.L4.FitPar.mp=(aRun->getFitPar())[1];													     
       sdsDet->cSignalCorr.L4.FitPar.area=(aRun->getFitPar())[2];												     
       sdsDet->cSignalCorr.L4.FitPar.gsigma=(aRun->getFitPar())[3];												     
       sdsDet->cSignalCorr.L4.FitPar.mp_peak=(sdsDet->cSignalCorr.L4.FitPar.mp - sdsDet->cSignalCorr.L4.FitPar.peak);						     
       																				     
       sdsDet->cSignalCorr.L4.FitPar.peak=aRun->pLanConv[0];													     
       sdsDet->cSignalCorr.L4.FitPar.FWHM=aRun->pLanConv[1];													     
																				     
       sdsDet->cSignalCorr.L4.FitPar.ewidth=(aRun->getFitParErr())[0];												     
       sdsDet->cSignalCorr.L4.FitPar.emp=(aRun->getFitParErr())[1];												     
       sdsDet->cSignalCorr.L4.FitPar.earea=(aRun->getFitParErr())[2];												     
       sdsDet->cSignalCorr.L4.FitPar.egsigma=(aRun->getFitParErr())[3];												     
     }																				     
     																				     
     if (iDet==1||iDet==2) {																	     
       sdsDet->cSignalCorr.L5.HistoPar.entries=aRun->doHisto("Layer","cSignalCorr_",DetList,"Layer_5_onTrack");							     
       sdsDet->cSignalCorr.L5.HistoPar.mean=(aRun->getHistoPar())[0];												     
       sdsDet->cSignalCorr.L5.HistoPar.rms=(aRun->getHistoPar())[1];												     
       																				     
       sdsDet->cSignalCorr.L6.HistoPar.entries=aRun->doHisto("Layer","cSignalCorr_",DetList,"Layer_6_onTrack");							     
       sdsDet->cSignalCorr.L6.HistoPar.mean=(aRun->getHistoPar())[0];												     
       sdsDet->cSignalCorr.L6.HistoPar.rms=(aRun->getHistoPar())[1];												     
       //FiT																			     
       																				     
       sdsDet->cSignalCorr.L5.FitPar.entries=aRun->doFit(runList,"Layer","cSignalCorr_",DetList,"Layer_5_onTrack");						     
       sdsDet->cSignalCorr.L5.FitPar.width=(aRun->getFitPar())[0];												     
       sdsDet->cSignalCorr.L5.FitPar.mp=(aRun->getFitPar())[1];													     
       sdsDet->cSignalCorr.L5.FitPar.area=(aRun->getFitPar())[2];												     
       sdsDet->cSignalCorr.L5.FitPar.gsigma=(aRun->getFitPar())[3];												     
       sdsDet->cSignalCorr.L5.FitPar.mp_peak=(sdsDet->cSignalCorr.L5.FitPar.mp - sdsDet->cSignalCorr.L5.FitPar.peak);						     
       																				     
       sdsDet->cSignalCorr.L5.FitPar.peak=aRun->pLanConv[0];													     
       sdsDet->cSignalCorr.L5.FitPar.FWHM=aRun->pLanConv[1];													     
       																				     
       sdsDet->cSignalCorr.L5.FitPar.ewidth=(aRun->getFitParErr())[0];												     
       sdsDet->cSignalCorr.L5.FitPar.emp=(aRun->getFitParErr())[1];												     
       sdsDet->cSignalCorr.L5.FitPar.earea=(aRun->getFitParErr())[2];												     
       sdsDet->cSignalCorr.L5.FitPar.egsigma=(aRun->getFitParErr())[3];												     
       																				     
       sdsDet->cSignalCorr.L6.FitPar.entries=aRun->doFit(runList,"Layer","cSignalCorr_",DetList,"Layer_6_onTrack");						     
       sdsDet->cSignalCorr.L6.FitPar.width=(aRun->getFitPar())[0];												     
       sdsDet->cSignalCorr.L6.FitPar.mp=(aRun->getFitPar())[1];													     
       sdsDet->cSignalCorr.L6.FitPar.area=(aRun->getFitPar())[2];												     
       sdsDet->cSignalCorr.L6.FitPar.gsigma=(aRun->getFitPar())[3];												     
       sdsDet->cSignalCorr.L6.FitPar.mp_peak=(sdsDet->cSignalCorr.L6.FitPar.mp - sdsDet->cSignalCorr.L6.FitPar.peak);						     
       																				     
       sdsDet->cSignalCorr.L6.FitPar.peak=aRun->pLanConv[0];													     
       sdsDet->cSignalCorr.L6.FitPar.FWHM=aRun->pLanConv[1];													     
       																				     
       sdsDet->cSignalCorr.L6.FitPar.ewidth=(aRun->getFitParErr())[0];												     
       sdsDet->cSignalCorr.L6.FitPar.emp=(aRun->getFitParErr())[1];												     
       sdsDet->cSignalCorr.L6.FitPar.earea=(aRun->getFitParErr())[2];												     
       sdsDet->cSignalCorr.L6.FitPar.egsigma=(aRun->getFitParErr())[3];												     
     }																				     
     if (iDet==2){																		     
       sdsDet->cSignalCorr.L7.HistoPar.entries=aRun->doHisto("Layer","cSignalCorr_",DetList,"Layer_7_onTrack");							     
       sdsDet->cSignalCorr.L7.HistoPar.mean=(aRun->getHistoPar())[0];												     
       sdsDet->cSignalCorr.L7.HistoPar.rms=(aRun->getHistoPar())[1];												     
       																				     
       sdsDet->cSignalCorr.L8.HistoPar.entries=aRun->doHisto("Layer","cSignalCorr_",DetList,"Layer_8_onTrack");							     
       sdsDet->cSignalCorr.L8.HistoPar.mean=(aRun->getHistoPar())[0];												     
       sdsDet->cSignalCorr.L8.HistoPar.rms=(aRun->getHistoPar())[1];												     
       																				     
       sdsDet->cSignalCorr.L9.HistoPar.entries=aRun->doHisto("Layer","cSignalCorr_",DetList,"Layer_9_onTrack");							     
       sdsDet->cSignalCorr.L9.HistoPar.mean=(aRun->getHistoPar())[0];												     
       sdsDet->cSignalCorr.L9.HistoPar.rms=(aRun->getHistoPar())[1];												     
       																				     
       sdsDet->cSignalCorr.L7.FitPar.entries=aRun->doFit(runList,"Layer","cSignalCorr_",DetList,"Layer_7_onTrack");						     
       sdsDet->cSignalCorr.L7.FitPar.width=(aRun->getFitPar())[0];												     
       sdsDet->cSignalCorr.L7.FitPar.mp=(aRun->getFitPar())[1];													     
       sdsDet->cSignalCorr.L7.FitPar.area=(aRun->getFitPar())[2];												     
       sdsDet->cSignalCorr.L7.FitPar.gsigma=(aRun->getFitPar())[3];												     
       sdsDet->cSignalCorr.L7.FitPar.mp_peak=(sdsDet->cSignalCorr.L7.FitPar.mp - sdsDet->cSignalCorr.L7.FitPar.peak);						     
       																				     
       sdsDet->cSignalCorr.L7.FitPar.peak=aRun->pLanConv[0];													     
       sdsDet->cSignalCorr.L7.FitPar.FWHM=aRun->pLanConv[1];													     
       																				     
       sdsDet->cSignalCorr.L7.FitPar.ewidth=(aRun->getFitParErr())[0];												     
       sdsDet->cSignalCorr.L7.FitPar.emp=(aRun->getFitParErr())[1];												     
       sdsDet->cSignalCorr.L7.FitPar.earea=(aRun->getFitParErr())[2];												     
       sdsDet->cSignalCorr.L7.FitPar.egsigma=(aRun->getFitParErr())[3];												     
       																				     
       sdsDet->cSignalCorr.L8.FitPar.entries=aRun->doFit(runList,"Layer","cSignalCorr_",DetList,"Layer_8_onTrack");						     
       sdsDet->cSignalCorr.L8.FitPar.width=(aRun->getFitPar())[0];												     
       sdsDet->cSignalCorr.L8.FitPar.mp=(aRun->getFitPar())[1];													     
       sdsDet->cSignalCorr.L8.FitPar.area=(aRun->getFitPar())[2];												     
       sdsDet->cSignalCorr.L8.FitPar.gsigma=(aRun->getFitPar())[3];												     
       sdsDet->cSignalCorr.L8.FitPar.mp_peak=(sdsDet->cSignalCorr.L8.FitPar.mp - sdsDet->cSignalCorr.L8.FitPar.peak);						     
       																				     
       sdsDet->cSignalCorr.L8.FitPar.peak=aRun->pLanConv[0];													     
       sdsDet->cSignalCorr.L8.FitPar.FWHM=aRun->pLanConv[1];													     
       																				     
       sdsDet->cSignalCorr.L8.FitPar.ewidth=(aRun->getFitParErr())[0];												     
       sdsDet->cSignalCorr.L8.FitPar.emp=(aRun->getFitParErr())[1];												     
       sdsDet->cSignalCorr.L8.FitPar.earea=(aRun->getFitParErr())[2];												     
       sdsDet->cSignalCorr.L8.FitPar.egsigma=(aRun->getFitParErr())[3];												     
       																				     
       sdsDet->cSignalCorr.L9.FitPar.entries=aRun->doFit(runList,"Layer","cSignalCorr_",DetList,"Layer_9_onTrack");						     
       sdsDet->cSignalCorr.L9.FitPar.width=(aRun->getFitPar())[0];												     
       sdsDet->cSignalCorr.L9.FitPar.mp=(aRun->getFitPar())[1];													     
       sdsDet->cSignalCorr.L9.FitPar.area=(aRun->getFitPar())[2];												     
       sdsDet->cSignalCorr.L9.FitPar.gsigma=(aRun->getFitPar())[3];												     
       sdsDet->cSignalCorr.L9.FitPar.mp_peak=(sdsDet->cSignalCorr.L9.FitPar.mp - sdsDet->cSignalCorr.L9.FitPar.peak);						     
       																				     
       sdsDet->cSignalCorr.L9.FitPar.peak=aRun->pLanConv[0];													     
       sdsDet->cSignalCorr.L9.FitPar.FWHM=aRun->pLanConv[1];													     
       																				     
       sdsDet->cSignalCorr.L9.FitPar.ewidth=(aRun->getFitParErr())[0];												     
       sdsDet->cSignalCorr.L9.FitPar.emp=(aRun->getFitParErr())[1];												     
       sdsDet->cSignalCorr.L9.FitPar.earea=(aRun->getFitParErr())[2];												     
       sdsDet->cSignalCorr.L9.FitPar.egsigma=(aRun->getFitParErr())[3];												     
     }																				     
   }																				     
     																				     
     void StoNCorrLayer(Int_t iDet,int runList, TIFRun* aRun,FinalSummary_t* sdsDet,char* DetList){								     
       cout << "...StoNCorr Called..." << endl;															     
       sdsDet->cStoNCorr.L1.HistoPar.entries=aRun->doHisto("Layer","cStoNCorr_",DetList,"Layer_1_onTrack");							     
       sdsDet->cStoNCorr.L1.HistoPar.mean=(aRun->getHistoPar())[0];												     
       sdsDet->cStoNCorr.L1.HistoPar.rms=(aRun->getHistoPar())[1];												     
       																				     
       sdsDet->cStoNCorr.L2.HistoPar.entries=aRun->doHisto("Layer","cStoNCorr_",DetList,"Layer_2_onTrack");							     
       sdsDet->cStoNCorr.L2.HistoPar.mean=(aRun->getHistoPar())[0];												     
       sdsDet->cStoNCorr.L2.HistoPar.rms=(aRun->getHistoPar())[1];												     
       																				     
       sdsDet->cStoNCorr.L3.HistoPar.entries=aRun->doHisto("Layer","cStoNCorr_",DetList,"Layer_3_onTrack");							     
       sdsDet->cStoNCorr.L3.HistoPar.mean=(aRun->getHistoPar())[0];												     
       sdsDet->cStoNCorr.L3.HistoPar.rms=(aRun->getHistoPar())[1];												     
       //cout << "====================FIT==============" << endl;												     
       																				     
       sdsDet->cStoNCorr.L1.FitPar.entries=aRun->doFit(runList,"Layer","cStoNCorr_",DetList,"Layer_1_onTrack");							     
       sdsDet->cStoNCorr.L1.FitPar.width=(aRun->getFitPar())[0];												     
       sdsDet->cStoNCorr.L1.FitPar.mp=(aRun->getFitPar())[1];													     
       sdsDet->cStoNCorr.L1.FitPar.area=(aRun->getFitPar())[2];													     
       sdsDet->cStoNCorr.L1.FitPar.gsigma=(aRun->getFitPar())[3];												     
       sdsDet->cStoNCorr.L1.FitPar.mp_peak=(sdsDet->cStoNCorr.L1.FitPar.mp - sdsDet->cStoNCorr.L1.FitPar.peak);							     
       																				     
       sdsDet->cStoNCorr.L1.FitPar.peak=aRun->pLanConv[0];													     
       sdsDet->cStoNCorr.L1.FitPar.FWHM=aRun->pLanConv[1];													     
       																				     
       sdsDet->cStoNCorr.L1.FitPar.ewidth=(aRun->getFitParErr())[0];												     
       sdsDet->cStoNCorr.L1.FitPar.emp=(aRun->getFitParErr())[1];												     
       sdsDet->cStoNCorr.L1.FitPar.earea=(aRun->getFitParErr())[2];												     
       sdsDet->cStoNCorr.L1.FitPar.egsigma=(aRun->getFitParErr())[3];												     
       																				     
       sdsDet->cStoNCorr.L2.FitPar.entries=aRun->doFit(runList,"Layer","cStoNCorr_",DetList,"Layer_2_onTrack");							     
       sdsDet->cStoNCorr.L2.FitPar.width=(aRun->getFitPar())[0];												     
       sdsDet->cStoNCorr.L2.FitPar.mp=(aRun->getFitPar())[1];													     
       sdsDet->cStoNCorr.L2.FitPar.area=(aRun->getFitPar())[2];													     
       sdsDet->cStoNCorr.L2.FitPar.gsigma=(aRun->getFitPar())[3];												     
       sdsDet->cStoNCorr.L2.FitPar.mp_peak=(sdsDet->cStoNCorr.L2.FitPar.mp - sdsDet->cStoNCorr.L2.FitPar.peak);							     
       																				     
       sdsDet->cStoNCorr.L2.FitPar.peak=aRun->pLanConv[0];													     
       sdsDet->cStoNCorr.L2.FitPar.FWHM=aRun->pLanConv[1];													     
       																				     
       sdsDet->cStoNCorr.L2.FitPar.ewidth=(aRun->getFitParErr())[0];												     
       sdsDet->cStoNCorr.L2.FitPar.emp=(aRun->getFitParErr())[1];												     
       sdsDet->cStoNCorr.L2.FitPar.earea=(aRun->getFitParErr())[2];												     
       sdsDet->cStoNCorr.L2.FitPar.egsigma=(aRun->getFitParErr())[3];												     
       																				     
       sdsDet->cStoNCorr.L3.FitPar.entries=aRun->doFit(runList,"Layer","cStoNCorr_",DetList,"Layer_3_onTrack");							     
       sdsDet->cStoNCorr.L3.FitPar.width=(aRun->getFitPar())[0];												     
       sdsDet->cStoNCorr.L3.FitPar.mp=(aRun->getFitPar())[1];													     
       sdsDet->cStoNCorr.L3.FitPar.area=(aRun->getFitPar())[2];													     
       sdsDet->cStoNCorr.L3.FitPar.gsigma=(aRun->getFitPar())[3];												     
       sdsDet->cStoNCorr.L3.FitPar.mp_peak=(sdsDet->cStoNCorr.L3.FitPar.mp - sdsDet->cStoNCorr.L3.FitPar.peak);							     
       																				     
       sdsDet->cStoNCorr.L3.FitPar.peak=aRun->pLanConv[0];													     
       sdsDet->cStoNCorr.L3.FitPar.FWHM=aRun->pLanConv[1];													     
       																				     
       sdsDet->cStoNCorr.L3.FitPar.ewidth=(aRun->getFitParErr())[0];												     
       sdsDet->cStoNCorr.L3.FitPar.emp=(aRun->getFitParErr())[1];												     
       sdsDet->cStoNCorr.L3.FitPar.earea=(aRun->getFitParErr())[2];												     
       sdsDet->cStoNCorr.L3.FitPar.egsigma=(aRun->getFitParErr())[3];												     
       																				     
       if (iDet !=3) {																		     
	 sdsDet->cStoNCorr.L4.HistoPar.entries=aRun->doHisto("Layer","cStoNCorr_",DetList,"Layer_4_onTrack");							     
	 sdsDet->cStoNCorr.L4.HistoPar.mean=(aRun->getHistoPar())[0];												     
	 sdsDet->cStoNCorr.L4.HistoPar.rms=(aRun->getHistoPar())[1];												     
	 cout <<"============FIT=============" << endl;														     
	 sdsDet->cStoNCorr.L4.FitPar.entries=aRun->doFit(runList,"Layer","cStoNCorr_",DetList,"Layer_4_onTrack");						     
	 sdsDet->cStoNCorr.L4.FitPar.width=(aRun->getFitPar())[0];												     
	 sdsDet->cStoNCorr.L4.FitPar.mp=(aRun->getFitPar())[1];													     
	 sdsDet->cStoNCorr.L4.FitPar.area=(aRun->getFitPar())[2];												     
	 sdsDet->cStoNCorr.L4.FitPar.gsigma=(aRun->getFitPar())[3];												     
	 sdsDet->cStoNCorr.L4.FitPar.mp_peak=(sdsDet->cStoNCorr.L4.FitPar.mp - sdsDet->cStoNCorr.L4.FitPar.peak);						     
	 																			     
	 sdsDet->cStoNCorr.L4.FitPar.peak=aRun->pLanConv[0];													     
	 sdsDet->cStoNCorr.L4.FitPar.FWHM=aRun->pLanConv[1];													     
	 																			     
	 sdsDet->cStoNCorr.L4.FitPar.ewidth=(aRun->getFitParErr())[0];												     
	 sdsDet->cStoNCorr.L4.FitPar.emp=(aRun->getFitParErr())[1];												     
	 sdsDet->cStoNCorr.L4.FitPar.earea=(aRun->getFitParErr())[2];												     
	 sdsDet->cStoNCorr.L4.FitPar.egsigma=(aRun->getFitParErr())[3];												     
       }																			     
       																				     
       if (iDet==1||iDet==2) {																	     
	 sdsDet->cStoNCorr.L5.HistoPar.entries=aRun->doHisto("Layer","cStoNCorr_",DetList,"Layer_5_onTrack");							     
	 sdsDet->cStoNCorr.L5.HistoPar.mean=(aRun->getHistoPar())[0];												     
	 sdsDet->cStoNCorr.L5.HistoPar.rms=(aRun->getHistoPar())[1];												     
	 																			     
	 sdsDet->cStoNCorr.L6.HistoPar.entries=aRun->doHisto("Layer","cStoNCorr_",DetList,"Layer_6_onTrack");							     
	 sdsDet->cStoNCorr.L6.HistoPar.mean=(aRun->getHistoPar())[0];												     
	 sdsDet->cStoNCorr.L6.HistoPar.rms=(aRun->getHistoPar())[1];												     
	 //Fit																			     
	 sdsDet->cStoNCorr.L5.FitPar.entries=aRun->doFit(runList,"Layer","cStoNCorr_",DetList,"Layer_5_onTrack");						     
	 sdsDet->cStoNCorr.L5.FitPar.width=(aRun->getFitPar())[0];												     
	 sdsDet->cStoNCorr.L5.FitPar.mp=(aRun->getFitPar())[1];													     
	 sdsDet->cStoNCorr.L5.FitPar.area=(aRun->getFitPar())[2];												     
	 sdsDet->cStoNCorr.L5.FitPar.gsigma=(aRun->getFitPar())[3];												     
	 sdsDet->cStoNCorr.L5.FitPar.mp_peak=(sdsDet->cStoNCorr.L5.FitPar.mp - sdsDet->cStoNCorr.L5.FitPar.peak);						     
	 																			     
	 sdsDet->cStoNCorr.L5.FitPar.peak=aRun->pLanConv[0];													     
	 sdsDet->cStoNCorr.L5.FitPar.FWHM=aRun->pLanConv[1];													     
	 																			     
	 sdsDet->cStoNCorr.L5.FitPar.ewidth=(aRun->getFitParErr())[0];												     
	 sdsDet->cStoNCorr.L5.FitPar.emp=(aRun->getFitParErr())[1];												     
	 sdsDet->cStoNCorr.L5.FitPar.earea=(aRun->getFitParErr())[2];												     
	 sdsDet->cStoNCorr.L5.FitPar.egsigma=(aRun->getFitParErr())[3];												     
	 																			     
	 sdsDet->cStoNCorr.L6.FitPar.entries=aRun->doFit(runList,"Layer","cStoNCorr_",DetList,"Layer_6_onTrack");						     
	 sdsDet->cStoNCorr.L6.FitPar.width=(aRun->getFitPar())[0];												     
	 sdsDet->cStoNCorr.L6.FitPar.mp=(aRun->getFitPar())[1];													     
	 sdsDet->cStoNCorr.L6.FitPar.area=(aRun->getFitPar())[2];												     
	 sdsDet->cStoNCorr.L6.FitPar.gsigma=(aRun->getFitPar())[3];												     
	 sdsDet->cStoNCorr.L6.FitPar.mp_peak=(sdsDet->cStoNCorr.L6.FitPar.mp - sdsDet->cStoNCorr.L6.FitPar.peak);						     
	 																			     
	 sdsDet->cStoNCorr.L6.FitPar.peak=aRun->pLanConv[0];													     
	 sdsDet->cStoNCorr.L6.FitPar.FWHM=aRun->pLanConv[1];													     
	 																			     
	 sdsDet->cStoNCorr.L6.FitPar.ewidth=(aRun->getFitParErr())[0];												     
	 sdsDet->cStoNCorr.L6.FitPar.emp=(aRun->getFitParErr())[1];												     
	 sdsDet->cStoNCorr.L6.FitPar.earea=(aRun->getFitParErr())[2];												     
	 sdsDet->cStoNCorr.L6.FitPar.egsigma=(aRun->getFitParErr())[3];												     
       }																			     
       if (iDet==2){																		     
	 sdsDet->cStoNCorr.L7.HistoPar.entries=aRun->doHisto("Layer","cStoNCorr_",DetList,"Layer_7_onTrack");							     
	 sdsDet->cStoNCorr.L7.HistoPar.mean=(aRun->getHistoPar())[0];												     
	 sdsDet->cStoNCorr.L7.HistoPar.rms=(aRun->getHistoPar())[1];												     
	 																			     
	 sdsDet->cStoNCorr.L8.HistoPar.entries=aRun->doHisto("Layer","cStoNCorr_",DetList,"Layer_8_onTrack");							     
	 sdsDet->cStoNCorr.L8.HistoPar.mean=(aRun->getHistoPar())[0];												     
	 sdsDet->cStoNCorr.L8.HistoPar.rms=(aRun->getHistoPar())[1];												     
	 																			     
	 sdsDet->cStoNCorr.L9.HistoPar.entries=aRun->doHisto("Layer","cStoNCorr_",DetList,"Layer_9_onTrack");							     
	 sdsDet->cStoNCorr.L9.HistoPar.mean=(aRun->getHistoPar())[0];												     
	 sdsDet->cStoNCorr.L9.HistoPar.rms=(aRun->getHistoPar())[1];												     
	 																			     
	 //Fit																			     
	 sdsDet->cStoNCorr.L7.FitPar.entries=aRun->doFit(runList,"Layer","cStoNCorr_",DetList,"Layer_7_onTrack");						     
	 sdsDet->cStoNCorr.L7.FitPar.width=(aRun->getFitPar())[0];												     
	 sdsDet->cStoNCorr.L7.FitPar.mp=(aRun->getFitPar())[1];													     
	 sdsDet->cStoNCorr.L7.FitPar.area=(aRun->getFitPar())[2];												     
	 sdsDet->cStoNCorr.L7.FitPar.gsigma=(aRun->getFitPar())[3];												     
	 sdsDet->cStoNCorr.L7.FitPar.mp_peak=(sdsDet->cStoNCorr.L7.FitPar.mp - sdsDet->cStoNCorr.L7.FitPar.peak);						     
	 																			     
	 sdsDet->cStoNCorr.L7.FitPar.peak=aRun->pLanConv[0];													     
	 sdsDet->cStoNCorr.L7.FitPar.FWHM=aRun->pLanConv[1];													     
	 																			     
	 sdsDet->cStoNCorr.L7.FitPar.ewidth=(aRun->getFitParErr())[0];												     
	 sdsDet->cStoNCorr.L7.FitPar.emp=(aRun->getFitParErr())[1];												     
	 sdsDet->cStoNCorr.L7.FitPar.earea=(aRun->getFitParErr())[2];												     
	 sdsDet->cStoNCorr.L7.FitPar.egsigma=(aRun->getFitParErr())[3];												     
	 																			     
	 sdsDet->cStoNCorr.L8.FitPar.entries=aRun->doFit(runList,"Layer","cStoNCorr_",DetList,"Layer_8_onTrack");						     
	 sdsDet->cStoNCorr.L8.FitPar.width=(aRun->getFitPar())[0];												     
	 sdsDet->cStoNCorr.L8.FitPar.mp=(aRun->getFitPar())[1];													     
	 sdsDet->cStoNCorr.L8.FitPar.area=(aRun->getFitPar())[2];												     
	 sdsDet->cStoNCorr.L8.FitPar.gsigma=(aRun->getFitPar())[3];												     
	 sdsDet->cStoNCorr.L8.FitPar.mp_peak=(sdsDet->cStoNCorr.L8.FitPar.mp - sdsDet->cStoNCorr.L8.FitPar.peak);						     
	 																			     
	 sdsDet->cStoNCorr.L8.FitPar.peak=aRun->pLanConv[0];													     
	 sdsDet->cStoNCorr.L8.FitPar.FWHM=aRun->pLanConv[1];													     
	 																			     
	 sdsDet->cStoNCorr.L8.FitPar.ewidth=(aRun->getFitParErr())[0];												     
	 sdsDet->cStoNCorr.L8.FitPar.emp=(aRun->getFitParErr())[1];												     
	 sdsDet->cStoNCorr.L8.FitPar.earea=(aRun->getFitParErr())[2];												     
	 sdsDet->cStoNCorr.L8.FitPar.egsigma=(aRun->getFitParErr())[3];												     
	 																			     
	 sdsDet->cStoNCorr.L9.FitPar.entries=aRun->doFit(runList,"Layer","cStoNCorr_",DetList,"Layer_9_onTrack");						     
	 sdsDet->cStoNCorr.L9.FitPar.width=(aRun->getFitPar())[0];												     
	 sdsDet->cStoNCorr.L9.FitPar.mp=(aRun->getFitPar())[1];													     
	 sdsDet->cStoNCorr.L9.FitPar.area=(aRun->getFitPar())[2];												     
	 sdsDet->cStoNCorr.L9.FitPar.gsigma=(aRun->getFitPar())[3];												     
	 sdsDet->cStoNCorr.L9.FitPar.mp_peak=(sdsDet->cStoNCorr.L9.FitPar.mp - sdsDet->cStoNCorr.L9.FitPar.peak);						     
	 																			     
	 sdsDet->cStoNCorr.L9.FitPar.peak=aRun->pLanConv[0];													     
	 sdsDet->cStoNCorr.L9.FitPar.FWHM=aRun->pLanConv[1];													     
	 																			     
	 sdsDet->cStoNCorr.L9.FitPar.ewidth=(aRun->getFitParErr())[0];												     
	 sdsDet->cStoNCorr.L9.FitPar.emp=(aRun->getFitParErr())[1];												     
	 sdsDet->cStoNCorr.L9.FitPar.earea=(aRun->getFitParErr())[2];												     
	 sdsDet->cStoNCorr.L9.FitPar.egsigma=(aRun->getFitParErr())[3];												     
	 																			     
       }																			     
     }                                                                                                                                                             

                                                                                                                             
