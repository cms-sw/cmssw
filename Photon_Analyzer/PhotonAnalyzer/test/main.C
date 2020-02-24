#include "PhotonClass.C"
#include "TROOT.h"
#include <algorithm>

//int main(int argc, const char *argv[]){
//  gROOT->ProcessLine("#include <vector>");
//  string outfile = argv[2];
//  long int nTotEvt = atof(argv[3]);
//  long int nPrintEvt = atof(argv[4]);
//  MyClass m(argv[1]);
//  m.Loop(outfile,nTotEvt,nPrintEvt);


int main(int argc, const char *argv[]){
  gROOT->ProcessLine("#include <vector>");
  string outfile = argv[2];
  long int nTotalEvent = atof(argv[3]);
  long int nPrintEvent = atof(argv[4]);
  TString S = (argv[1]);//"root://cmsxrootd.fnal.gov///store/user/sghosh/SINGLEELECTRON/SEv4p2.root";
  PhotonClass m(S);
  m.Loop(outfile,nTotalEvent,nPrintEvent);
  

}
