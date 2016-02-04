#include <stdlib.h>

#include "Riostream.h"
#include "TROOT.h"
#include "TTree.h"
#include "TChain.h"
#include "TMath.h"
#include "TRef.h"
#include "TRefArray.h"
#include "TH1.h"
#include "TH2.h"
#include "TDatabasePDG.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "UEAnalysisOnRootple.h"
using namespace std;

int main(int argc, char* argv[]) {
  
  char* filelist = argv[1];
  char* outname = argv[2];
  char* type = argv[3];
  char* jetStream = argv[4];
  Float_t lumi=atof(argv[5]);
  Float_t eta=atof(argv[6]);
  Float_t triggerPt=atof(argv[7]);
  char* tkCut= argv[8];
  Float_t ptCut = atof(argv[9]);

  string Trigger(jetStream);
  string AnalysisType(type);
  string TracksPt(tkCut);

  Float_t weight[7];

  Float_t Jet20_900[7]={1.000000,0.254008,0.027852,0.011589,0.002788,0.000531,0.000071};
  Float_t Jet20GlobalWeight_900=9.95;
  Float_t Jet60_900[7]={1.000000,0.416103,0.100088,0.019064,0.002536,0.0,0.0};
  Float_t Jet60GlobalWeight_900=1.86;
  Float_t Jet120_900[7]={1.000000,0.190474,0.025338,0.0,0.0,0.0,0.0};
  Float_t Jet120GlobalWeight_900=23.43;
  Float_t MB_900[7]={1.0,0.0,0.0,0.0,0.0,0.0,0.0};
  Float_t MBGlobalWeight_900=0.104;

  Float_t Jet20_500[7]={1.000000,0.263501,0.026244,0.010588,0.002454,0.000493,0.000067};
  Float_t Jet20GlobalWeight_500=11.32;
  Float_t Jet60_500[7]={1.000000,0.403430,0.093489,0.018767,0.002535,0.0,0.0};
  Float_t Jet60GlobalWeight_500=1.99;
  Float_t Jet120_500[7]={1.000000,0.200737,0.027111,0.0,0.0,0.0,0.0};
  Float_t Jet120GlobalWeight_500=23.43;
  Float_t MB_500[7]={1.0,0.0,0.0,0.0,0.0,0.0,0.0};
  Float_t MBGlobalWeight_500=0.276;

  if(TracksPt=="900"){
    if(Trigger=="Jet20"){
      for(int i=0;i<7;i++){
	weight[i]=Jet20_900[i]*Jet20GlobalWeight_900*lumi*0.1; 
	}
    }else if(Trigger=="Jet60"){
      for(int i=0;i<7;i++)
	weight[i]=Jet60_900[i]*Jet60GlobalWeight_900*lumi*0.1; 
    }else if(Trigger=="Jet120"){
      for(int i=0;i<7;i++)
	weight[i]=Jet120_900[i]*Jet120GlobalWeight_900*lumi*0.1; 
    }else if(Trigger=="MB"){
      for(int i=0;i<7;i++)
	weight[i]=MB_900[i]*MBGlobalWeight_900*lumi*0.1; 
    }else{
      cout<<"Select an undefinde Jet Stream "<<Trigger<<endl;
    }
  }

  if(TracksPt=="500"){
    if(Trigger=="Jet20"){
      for(int i=0;i<7;i++)
	weight[i]=Jet20_500[i]*Jet20GlobalWeight_500*lumi*0.1; 
    }else if(Trigger=="Jet60"){
      for(int i=0;i<7;i++)
	weight[i]=Jet60_500[i]*Jet60GlobalWeight_500*lumi*0.1; 
    }else if(Trigger=="Jet120"){
      for(int i=0;i<7;i++)
	weight[i]=Jet120_500[i]*Jet120GlobalWeight_500*lumi*0.1; 
    }else if(Trigger=="MB"){
      for(int i=0;i<7;i++)
	weight[i]=MB_500[i]*MBGlobalWeight_500*lumi*0.1; 
    }else{
      cout<<"Select an undefinde Jet Stream "<<Trigger<<endl;
    }
  }
  
  
  UEAnalysisOnRootple tt;
  
  tt.MultiAnalysis(filelist,outname,weight,eta,triggerPt,AnalysisType,Trigger,TracksPt,ptCut);
  
  cout << "end events loop" << endl;
  
  return 0;                                                                                                                                   
  
}

