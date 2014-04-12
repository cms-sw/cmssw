/*************************/
/*                       */
/* author: Pasquale Noli */
/* INFN Naples           */
/* Toy Montecarlo        */
/*                       */
/*************************/

//root include
#include "TRandom3.h"
#include "TH1.h"
#include "TF1.h"
#include "TFile.h"
#include "TDirectory.h"
#include <stdlib.h>
#include <vector>
#include <string>
#include <iostream>
#include <iterator>
#include <sstream>
#include <cmath>
#include <unistd.h>

using namespace std;

void fillRandom(int N, TH1F *pdf, TH1F * histo, double min, double max, TRandom3 * rndm){
  int i=0;
  double m=0;
  const double maxY = pdf->GetMaximum();
  const double nBins = pdf->GetNbinsX();
  const double xMin = pdf->GetXaxis()->GetXmin();
  const double xMax = pdf->GetXaxis()->GetXmax();
  do{
    m = rndm->Uniform(min, max);
    int n = (int)((m - xMin)/(xMax - xMin)*nBins) + 1;
    double y = pdf->GetBinContent(n);
    if(rndm->Uniform() < y/maxY){
      histo->Fill(m);
      i++;
    }
  }while(i<N);
} 


enum MuTag { globalMu, trackerMu, standaloneMu, undefinedMu };

MuTag mu(double effTrk, double effSa, TRandom3 * rndm) {
  double _isTraker     = rndm->Rndm();
  double _isStandAlone = rndm->Rndm();
  if((_isTraker< effTrk)   && (_isStandAlone< effSa)) return globalMu;
  if((_isStandAlone< effSa)&& (_isTraker >effTrk)) return standaloneMu;
  if((_isTraker < effTrk)  && (_isStandAlone > effSa)) return trackerMu;
  return undefinedMu;
}


bool efficiencyTag(double eff, TRandom3 * rndm ){
 return (rndm->Rndm()< eff);
}


class BkgShape {
public:
  BkgShape(double min, double max, double slope, double a0, double a1, double a2) :
    norm_(1), min_(min), max_(max), fmax_(0),
    slope_(slope), a0_(a0), a1_(a1), a2_(a2) { 
    normalize();
  }
  double operator()(double x) const {
    if(x < min_ || x > max_) return 0;
    return exp(-slope_*x)*(a0_ + (a1_ + a2_*x)*x);
  }
  double rndm(TRandom3 * rndm) const {
    double x, f;
    do {
      x = rndm->Uniform(min_, max_);
      f = operator()(x);
    } while(rndm->Uniform(0, fmax_) > f);
    return x;
  }
  double integral() const { return norm_; }

private:
  void normalize() {
    static const unsigned int steps = 1000;
    double s = 0, x, f;
    double base = max_ - min_;
    double dx = base/steps;
    for(unsigned int n = 0; n < steps; ++n) {
      x = min_ + (n * dx);
      s += (f = operator()(x))* dx;
      if(f > fmax_) fmax_ = f;
    }
    fmax_ *= 1.001;//max of f
    norm_ = s;
  }
  double norm_, min_, max_, fmax_;
  double slope_, a0_, a1_, a2_;
};

int main(int argc, char * argv[]){
  TRandom3 *rndm = new TRandom3();
  int o;
  char* endPtr;
  const char* pdf("analysis_Z_133pb_trackIso_3.root");
  double yield(3810.0), effTrk(.996), effSa(.987),effHlt(.913), effIso(.982),factor(1.0),MIN(60.),MAX(120.);
  double slopeMuTk(0.02), a0MuTk(1.0), a1MuTk(0.0), a2MuTk(0.0);
  double slopeMuMuNonIso(0.02),a0MuMuNonIso(1.0), a1MuMuNonIso(0.0), a2MuMuNonIso(0.0);
  double slopeMuSa(0.02), a0MuSa(1.0), a1MuSa(0.0), a2MuSa(0.0);  
// double yield(50550.0), effTrk(.998364), effSa(.989626),effHlt(.915496), effIso(.978575),factor(1.0);
  // double slopeMuTk(.015556), a0MuTk(.00035202), a1MuTk(2.99663), a2MuTk(-0.0211138);
  // double slopeMuMuNonIso(.0246876),a0MuMuNonIso(.884777), a1MuMuNonIso(6.67684), a2MuMuNonIso(-0.0523693);
  // BkgShape zMuTkBkgPdf(60., 120., slopeMuTk, a0MuTk, a1MuTk, a2MuTk);
  // BkgShape zMuMuNonIsoBkgPdf(60., 120., slopeMuMuNonIso, a0MuMuNonIso, a1MuMuNonIso, a2MuMuNonIso);

  int expt(1), seed(1);

  while ((o = getopt(argc, argv,"p:n:s:y:m:M:f:T:S:H:I:h"))!=EOF) {
    switch(o) {
    case 'p':
      pdf  = optarg;
      break;
    case 'n':
      expt  = strtoul(optarg,&endPtr,0);
      break;
    case 's':
      seed = strtoul(optarg,&endPtr,0);
      break;
    case 'y':
      yield = strtoul(optarg,&endPtr,0);
      break;
    case 'm':
      MIN = strtoul(optarg,&endPtr,0);
      break;
    case 'M':
      MAX = strtoul(optarg,&endPtr,0);
      break;
    case 'f':
      factor = strtoul(optarg,&endPtr,0);
      break;
    case 'T':
      effTrk  = strtod(optarg,&endPtr);
      break;
    case 'S':
      effSa = strtod(optarg,&endPtr);
      break;
    case 'H':
      effHlt = strtod(optarg,&endPtr);
      break;
    case 'I':
      effIso  = strtod(optarg,&endPtr);
      break;
    case 'h':
      cout<< " -p : input root file for pdf"<<endl <<" -n : number of experiment (default 1)"<<endl <<" -s : seed for generator (default 1)"<<endl <<" -T : efficiency of track (default 0.9984)"<<endl <<" -S : efficiency of standAlone(default 0.9896)"<< endl <<" -I : efficiency of Isolation (default 0.9786)" << endl << " -H : efficiency of HLT (default 0.9155)" <<endl << " -y : yield (default 50550)"<<endl<<" -f : scaling_factor for bkg (default 1.0)"<<endl<< " -m : Min (60)"<< endl<< " -M : Max (120)"<<endl;
      break;
    default:
      break;
    }
  }
  BkgShape zMuTkBkgPdf(MIN, MAX, slopeMuTk, a0MuTk, a1MuTk, a2MuTk);
  BkgShape zMuMuNonIsoBkgPdf(MIN, MAX, slopeMuMuNonIso, a0MuMuNonIso, a1MuMuNonIso, a2MuMuNonIso);
  BkgShape zMuSaBkgPdf(MIN, MAX, slopeMuSa, a0MuSa, a1MuSa, a2MuSa);
  MuTag mu1,mu2;
  rndm->SetSeed(seed);
  int count = 0; 
  //PDF
  TFile *inputfile = new TFile(pdf);
  TH1F *pdfzmm = (TH1F*)inputfile->Get("goodZToMuMuPlots/zMass");//pdf signal Zmumu(1hlt,2hlt), ZMuMunotIso, ZmuTk
  TH1F *pdfzmsa = (TH1F*)inputfile->Get("zmumuSaMassHistogram/zMass");//pdf signal ZmuSa
  double IntegralzmumuNoIsobkg =factor * ( zMuMuNonIsoBkgPdf.integral());
  double Integralzmutkbkg =factor * (zMuTkBkgPdf.integral());
  double Integralzmusabkg =factor * (zMuSaBkgPdf.integral());

  for(int j = 1; j <=expt; ++j){//loop on number of experiments  
    int N0 = rndm->Poisson(yield);
    int nMuTkBkg = rndm->Poisson(Integralzmutkbkg);
    int nMuMuNonIsoBkg = rndm->Poisson(IntegralzmumuNoIsobkg);
    int nMuSaBkg = rndm->Poisson(Integralzmusabkg);
    int Nmumu = 0;
    int N2HLT = 0;
    int N1HLT = 0;
    int NISO = 0;
    int NSa = 0;
    int NTk = 0;
    for(int i = 0; i < N0; ++i){//loop on Z Yield
      mu1=mu(effTrk,effSa, rndm);
      mu2=mu(effTrk,effSa, rndm);
      bool iso1 =  efficiencyTag(effIso,rndm);
      bool iso2 =  efficiencyTag(effIso,rndm);
      bool trig1 = efficiencyTag(effHlt,rndm);
      bool trig2 = efficiencyTag(effHlt,rndm);
   
      if(mu1 == globalMu && mu2 == globalMu){
	 if(iso1 && iso2){//two global mu isolated
	   if(trig1 && trig2) N2HLT++;//two trigger
	   else if((trig1 && !trig2)||(!trig1 && trig2)) N1HLT++;//one trigger
	 }
	 else if(!iso1 || !iso2){//at least one not iso
	   if( trig1 || trig2) NISO++;//at least one trigger
	 }
       }//end global
       else if(((mu1 == globalMu && trig1) &&  mu2 == standaloneMu ) 
	       ||((mu2 == globalMu && trig2) && mu1 == standaloneMu )){
	 if(iso1 && iso2) NSa++;
       }//end mu sa
      else if(((mu1 == globalMu && trig1) && mu2 == trackerMu) 
	       || ((mu2 == globalMu && trig2) && mu1 == trackerMu)){
	 if(iso1 && iso2) NTk++;
       }//end mu tk
     

    }//end of generation given the yield
       
    Nmumu = N2HLT + N1HLT;
    
    //Define signal Histo
    TH1F *zMuMu = new TH1F("zMass","zMass",200,0,200);
    TH1F *zMuMu2HLT = new TH1F("zMass","zMass",200,0,200);
    TH1F *zMuMu1HLT = new TH1F("zMass","zMass",200,0,200);
    TH1F *zMuMuNotIso= new TH1F("zMass","zMass",200,0,200);
    TH1F *zMuSa = new TH1F("zMass","zMass",200,0,200);
    TH1F *zMuTk = new TH1F("zMass","zMass",200,0,200);
    pdfzmsa->SetName("zMass");
  
    //Fill signal Histo
   
    fillRandom(Nmumu,pdfzmm,zMuMu,MIN,MAX, rndm);
    fillRandom(N2HLT, pdfzmm,zMuMu2HLT,MIN,MAX, rndm);
    fillRandom(N1HLT, pdfzmm,zMuMu1HLT,MIN,MAX, rndm);
    fillRandom(NISO,pdfzmm,zMuMuNotIso,MIN,MAX, rndm);
    fillRandom(NSa,pdfzmsa,zMuSa,MIN,MAX, rndm);
    fillRandom(NTk, pdfzmm,zMuTk,MIN,MAX, rndm);
        
    //output	
    char head[30];
    sprintf(head,"zmm_%d",j);
    string tail =".root";
    string title = head + tail;
    
    TFile *outputfile = new TFile(title.c_str(),"RECREATE");
    
    //Hierarchy directory  
    
    TDirectory * goodZToMuMu = outputfile->mkdir("goodZToMuMuPlots");
    TDirectory * goodZToMuMu2HLT = outputfile->mkdir("goodZToMuMu2HLTPlots");
    TDirectory * goodZToMuMu1HLT = outputfile->mkdir("goodZToMuMu1HLTPlots");
    TDirectory * nonIsolatedZToMuMu = outputfile->mkdir("nonIsolatedZToMuMuPlots");
    TDirectory * goodZToMuMuOneStandAloneMuon = outputfile->mkdir("goodZToMuMuOneStandAloneMuonPlots");
    TDirectory * zmumuSaMassHistogram = outputfile->mkdir("zmumuSaMassHistogram");
    TDirectory * goodZToMuMuOneTrack = outputfile->mkdir("goodZToMuMuOneTrackPlots");
  
    goodZToMuMu->cd();
    zMuMu->Write();
    
    goodZToMuMu2HLT->cd();
    zMuMu2HLT->Write();
    
    goodZToMuMu1HLT->cd();
    zMuMu1HLT->Write();
    
    nonIsolatedZToMuMu->cd();
    zMuMuNotIso->Write();
    
    goodZToMuMuOneStandAloneMuon->cd();
    zMuSa->Write();

    zmumuSaMassHistogram->cd();
    pdfzmsa->Write();


    goodZToMuMuOneTrack->cd();
    zMuTk->Write();
    
    
    outputfile->Write();
    outputfile->Close();
    
    delete zMuMu;
    delete zMuMu2HLT;
    delete zMuMu1HLT;
    delete zMuMuNotIso;
    delete zMuSa;
    delete zMuTk;
    
       
    
    //Define Background Histo
    TH1F *zMuMuBkg = new TH1F("zMass","zMass",200,0,200);
    TH1F *zMuMu2HLTBkg = new TH1F("zMass","zMass",200,0,200);
    TH1F *zMuMu1HLTBkg = new TH1F("zMass","zMass",200,0,200);
    TH1F *zMuSaBkg = new TH1F("zMass","zMass",200,0,200);
    TH1F *zMuSafromGoldenBkg = new TH1F("zMass","zMass",200,0,200);
    TH1F *zMuMuNotIsoBkg= new TH1F("zMass","zMass",200,0,200);
    TH1F *zMuTkBkg = new TH1F("zMass","zMass",200,0,200);
    
    
    
    //Fill >Bkg Histograms 
    for(int i = 0; i < nMuTkBkg; ++i) {
      zMuTkBkg->Fill(zMuTkBkgPdf.rndm(rndm));
    }
    for(int i = 0; i < nMuMuNonIsoBkg; ++i) {
      zMuMuNotIsoBkg->Fill(zMuMuNonIsoBkgPdf.rndm(rndm));
    }
    for(int i = 0; i < nMuSaBkg; ++i) {
      zMuSaBkg->Fill(zMuSaBkgPdf.rndm(rndm));
    }
    char head2[30];
    sprintf(head2,"bkg_%d",j);
    string title2 = head2 + tail;
    TFile *outputfile2 = new TFile(title2.c_str(),"RECREATE");
    
    //Hierarchy directory  
    TDirectory * goodZToMuMu2 = outputfile2->mkdir("goodZToMuMuPlots");
    TDirectory * goodZToMuMu2HLT2 = outputfile2->mkdir("goodZToMuMu2HLTPlots");
    TDirectory * goodZToMuMu1HLT2 = outputfile2->mkdir("goodZToMuMu1HLTPlots");
    TDirectory * nonIsolatedZToMuMu2 = outputfile2->mkdir("nonIsolatedZToMuMuPlots");
    TDirectory * goodZToMuMuOneStandAloneMuon2 = outputfile2->mkdir("goodZToMuMuOneStandAloneMuonPlots");
    TDirectory * zmumuSaMassHistogram2 = outputfile2->mkdir("zmumuSaMassHistogram");
    TDirectory * goodZToMuMuOneTrack2 = outputfile2->mkdir("goodZToMuMuOneTrackPlots");
    

    goodZToMuMu2->cd();
    zMuMuBkg->Write();
    
    goodZToMuMu2HLT2->cd();
    zMuMu2HLTBkg->Write();
    
    goodZToMuMu1HLT2->cd();
    zMuMu1HLTBkg->Write();
    
    nonIsolatedZToMuMu2->cd();
    zMuMuNotIsoBkg->Write();
    
    goodZToMuMuOneStandAloneMuon2->cd();
    zMuSaBkg->Write();

    zmumuSaMassHistogram2->cd();
    zMuSafromGoldenBkg->Write();

    goodZToMuMuOneTrack2->cd();
    zMuTkBkg->Write();
    
    outputfile2->Write();
    outputfile2->Close();

    delete zMuMuBkg;
    delete zMuMu2HLTBkg;
    delete zMuMu1HLTBkg;
    delete zMuMuNotIsoBkg;
    delete zMuSafromGoldenBkg;
    delete zMuSaBkg;
    delete zMuTkBkg;

    
    // cout<<count<<"\n";
    count++;
  }//end of experiments 

  delete inputfile;
    
  return 0;
  
}
