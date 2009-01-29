//root include
#include "TRandom3.h"
#include "TH1.h"
#include "TF1.h"
#include "TFile.h"
#include "TDirectory.h"
//std include
#include <vector>
#include <string>
#include <iostream>
#include <iterator>
#include <sstream>
#include <cmath>
using namespace std;

TRandom3 *eventGenerator = new TRandom3();

int o;
char* endPtr;
char* pdf("analysis_Z_133pb_trackIso_3.root");
double yield(50550),effTrk(.9883),effSa(.9896),effHlt(.9155),effIso(.9786);
int expt(1),seed(1);

void FillRandom(int N, TH1F *pdf, TH1F * histo){
  double m =0;
  for(int i =0 ; i <N ;++i){
    do{
      m=pdf->GetRandom();
      histo->Fill(m);
    }while(!(m>120 && m<60));
  }
 }

enum MuTag { globalMu, trackerMu, standaloneMu, undefinedMu };

MuTag mu(double effTrk, double effSa) {
  if( eventGenerator->Rndm()< effTrk && eventGenerator->Rndm()< effSa ){
    return globalMu;
  } else if(eventGenerator->Rndm()< effTrk){
    return trackerMu;
  }
  else if(eventGenerator->Rndm()< effSa){
    return standaloneMu;
  }
  else return undefinedMu;
}

double bkgShape(double x) {
  return  exp(-0.0223304 * x) * (0.000505041 + (0.019177 -0.00012970 * x) * x);
}

int main(int argc, char * argv[]){
  
  while ((o = getopt(argc, argv,"p:n:s:y:T:S:H:I:h"))!=EOF) {
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
      cout<< " -p : input root file for pdf"<<endl <<" -n : number of experiment (default 1)"<<endl <<" -s : seed for generator (default 1)"<<endl <<" -T : efficiency of track (default 0.9883)"<<endl <<" -S : efficiency of standAlone(default 0.9896)"<< endl <<" -I : efficiency of Isolation (default 0.9786)" << endl << " -H : efficiency of HLT (default 0.9155)" <<endl << " -y : yield (default 50550)"<<endl;
      break;
    default:
      break;
    }
  }

  MuTag mu1,mu2;
  eventGenerator->SetSeed(seed);
  int count = 0; 
  //PDF
  TFile *inputfile = new TFile(pdf);
  TH1F *pdfzmm = (TH1F*)inputfile->Get("goodZToMuMuPlots/zMass");//pdf signal Zmumu(1hlt,2hlt), ZMuMunotIso, ZmuTk
  TH1F *pdfzmsa = (TH1F*)inputfile->Get("zmumuSaMassHistogram/zMass");//pdf signal ZmuSa
  
  //pdf background
  TF1 * zmutkBkg  = new TF1("zmtkBkg","exp(-0.0223304 * x) * (0.000505041 + 0.019177 * x -0.00012970 * x * x)");//pdf StandAlone
  TF1 * zmunoisoBkg = new TF1("zmnoisoBkg","exp( -0.0232129* x) * ( 1.99999 + 0.0944887 * x -0.000859517 * x * x)");//pdf zmuNotIso
  

  for(int j = 0; j <expt; ++j){ 
    int N0 = eventGenerator->Poisson(yield);
    int Nmumu = 0;
    int N2HLT = 0;
    int N1HLT = 0;
    int NISO = 0;
    int NSa = 0;
    int NTk = 0;
    for(int i = 0; i < N0; ++i){
      mu1=mu(effTrk,effSa);
      mu2=mu(effTrk,effSa);
      
      double rHLT1 = eventGenerator->Rndm();
      double rISO1 = eventGenerator->Rndm();   
      double rHLT2 = eventGenerator->Rndm();
      double rISO2 = eventGenerator->Rndm();
      
      if(mu1 == globalMu && mu2 == globalMu){
	if(rISO1< effIso && rISO2 < effIso){//two global mu isolated
	  if(rHLT1< effHlt && rHLT2 < effHlt) N2HLT++;
	  else if((rHLT1< effHlt && !rHLT2 < effHlt)||(!rHLT1 < effHlt && rHLT2 < effHlt)) N1HLT++;
	} else if(!rISO1< effIso || !rISO2 < effIso){//at least one not iso
	  if( rHLT1 < effHlt || rHLT2 < effHlt) NISO++;
	}
      }else if((mu1 == globalMu && mu2 == trackerMu && rHLT1< effHlt ) || (mu2 == globalMu && mu1 == trackerMu && rHLT2< effHlt)){
	if(rISO1< effIso && rISO2 < effIso) NTk++;
      }else if((mu1 == globalMu && mu2 == standaloneMu && rHLT1< effHlt) ||(mu2 == globalMu && mu1 == standaloneMu && rHLT2< effHlt)){
	if(rISO1< effIso && rISO2 < effIso) NSa++;
      }
      
      Nmumu = N2HLT + N1HLT;
      
      //Define signal Histo
      TH1F *zMuMu = new TH1F("zMass_golden","zMass",200,0,200);
      TH1F *zMuMu2HLT = new TH1F("zMass_2hlt","zMass",200,0,200);
      TH1F *zMuMu1HLT = new TH1F("zMass_1hlt","zMass",200,0,200);
      TH1F *zMuMuNotIso= new TH1F("zMass_noIso","zMass",200,0,200);
      TH1F *zMuSa = new TH1F("zMass_sa","zMass",200,0,200);
      TH1F *zMuTk = new TH1F("zMass_tk","zMass",200,0,200);
      
      //Fill signal Histo
      FillRandom(Nmumu,pdfzmm,zMuMu );
      FillRandom(N2HLT, pdfzmm,zMuMu2HLT);
      FillRandom(N1HLT, pdfzmm,zMuMu1HLT);
      FillRandom(NISO,pdfzmm,zMuMuNotIso);
      FillRandom(NSa,pdfzmsa,zMuSa);
      FillRandom(NTk, pdfzmm,zMuTk);
            
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
      
      delete inputfile;
      // } //fine for
      
           
      //Define Background Histo
      TH1F *zMuMuBkg = new TH1F("zMass_golden","zMass",200,0,200);
      TH1F *zMuMu2HLTBkg = new TH1F("zMass_2hlt","zMass",200,0,200);
      TH1F *zMuMu1HLTBkg = new TH1F("zMass_1hlt","zMass",200,0,200);
      TH1F *zMuSaBkg = new TH1F("zMass_sa","zMass",200,0,200);
      TH1F *zMuMuNotIsoBkg= new TH1F("zMass_noIso","zMass",200,0,200);
      TH1F *zMuTkBkg = new TH1F("zMass_tk","zMass",200,0,200);
      
      int Nzmtbkg = eventGenerator->Poisson(1890);
      int NzmNoIsobkg = eventGenerator->Poisson(2600);
      //Fill >Bkg Histograms 
      zMuMuNotIsoBkg->FillRandom("zmnoisoBkg",Nzmtbkg);
      zMuTkBkg->FillRandom("zmtkBkg",Nzmtbkg );
      
      
      char head2[30];
      sprintf(head2,"bgk_%d",j);
      string title2 = head2 + tail;
      TFile *outputfile2 = new TFile(title2.c_str(),"RECREATE");
      
      //Hierarchy directory  
      TDirectory * goodZToMuMu2 = outputfile2->mkdir("goodZToMuMuPlots");
      TDirectory * goodZToMuMu2HLT2 = outputfile2->mkdir("goodZToMuMu2HLTPlots");
      TDirectory * goodZToMuMu1HLT2 = outputfile2->mkdir("goodZToMuMu1HLTPlots");
      TDirectory * nonIsolatedZToMuMu2 = outputfile2->mkdir("nonIsolatedZToMuMuPlots");
      TDirectory * goodZToMuMuOneStandAloneMuon2 = outputfile2->mkdir("goodZToMuMuOneStandAloneMuonPlots");
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
      
      goodZToMuMuOneTrack2->cd();
      zMuTkBkg->Write();
      
      
      delete zMuMuBkg;
      delete zMuMu2HLTBkg;
      delete zMuMu1HLTBkg;
      delete zMuMuNotIsoBkg;
      delete zMuSaBkg;
      delete zMuTkBkg;
    
    }//end of generation given yield
    cout<<count<<"\n";
    count++;
  }//end of experiment 
  return 0;
  
}
