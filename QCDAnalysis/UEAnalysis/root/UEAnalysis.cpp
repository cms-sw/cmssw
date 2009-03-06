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

  cout << "Running UEAnalysis..." << endl;
  
  if ( argc != 6 )
    {
      /// old command was
      ///
      ///  ./UEAnalysis list test.root UEGen MB 1 2 1500 1500
      ///
      /// new command is
      ///
      /// ./UEAnalysis list test.root UEGen 2 1500

      cout << "[Fatal Error] Wrong number of parameters. Please run by invoking" << endl;
      cout << "              ./UEAnalysis <input list> <output> <mode> <eta-range> <pT-threshold>" << endl;
      cout << "exitting ..." << endl;

      return 1;
    }

  char* filelist = argv[1];
  char* outname = argv[2];
  char* type = argv[3];
  char* jetStream = "";
  Float_t lumi= 1.;
  Float_t eta=atof(argv[4]);
  char* tkCut= argv[5];
  Float_t ptCut = atof(tkCut);

  cout << endl;
  cout << "\tinput list   " << filelist << endl;
  cout << "\toutput       " << outname << endl;
  cout << "\tmode         " << type << endl;
  cout << "\teta-range    -" << eta << " ... " << eta << endl;
  cout << "\tpT-threshold " << ptCut << " MeV/c" << endl;
  cout << endl;
  
  string Trigger(jetStream);
  string AnalysisType(type);
  string TracksPt(tkCut);


  UEAnalysisWeight ueW;
  //  std::vector<float> weight = ueW.calculate(TracksPt,Trigger,lumi);
  UEAnalysisOnRootple tt;

  // void UEAnalysisOnRootple::MultiAnalysis(char* filelist,char* outname,vector<float> weight,Float_t eta,
  //                                         string type,string trigger,string tkpt,Float_t ptCut)
  //
  tt.MultiAnalysis(filelist,outname,ueW.calculate(),eta,AnalysisType,Trigger,TracksPt,ptCut);
  
  cout << "end events loop" << endl;
  
  return 0;

}

