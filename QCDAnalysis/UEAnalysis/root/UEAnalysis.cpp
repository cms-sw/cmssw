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
#include "UEAnalysisWeight.h"

using namespace std;

int main(int argc, char* argv[]) {
  
  char* filelist = argv[1];
  char* outname = argv[2];
  char* type = argv[3];
  char* jetStream = argv[4];
  Float_t lumi=atof(argv[5]);
  Float_t eta=atof(argv[6]);
  char* tkCut= argv[7];
  Float_t ptCut = atof(argv[8]);

  string Trigger(jetStream);
  string AnalysisType(type);
  string TracksPt(tkCut);


  UEAnalysisWeight ueW;
  //  std::vector<float> weight = ueW.calculate(TracksPt,Trigger,lumi);
  UEAnalysisOnRootple tt;

  tt.MultiAnalysis(filelist,outname,ueW.calculate(),eta,AnalysisType,Trigger,TracksPt,ptCut);
  
  cout << "end events loop" << endl;
  
  return 0;

}

