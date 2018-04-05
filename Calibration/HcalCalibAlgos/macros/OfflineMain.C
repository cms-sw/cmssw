#include "HBHEMuonOfflineAnalyzer.C"
#include <TStyle.h>
#include <TCanvas.h>
#include <iostream>
#include <fstream>
#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TProfile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <vector>
#include <TTree.h>
#include <string>
#include "TString.h"
using namespace std;

Bool_t FillChain(TChain *chain, const char* inputFileList);

int main(Int_t argc, Char_t *argv[]) {

  if (argc<4) {
    std::cerr << "Please give 3 arguments \n"
	      << "InputFileName" << "\n"
	      << "outputFileName" << "\n"
	      << "processName" << "\n"
	      << std::endl;
    return -1;
  }

  const char *inputFileList   = argv[1];
  const char *outFileName     = argv[2];
  const char *processName     = argv[3];
  std::cout << "Input File List "  << inputFileList << std::endl
	    << "Output FIle Name " << outFileName << std::endl
	    << "Process Name "     << processName << std::endl;

  int flag(0), mode(4), maxDHB(5), maxDHE(7), runLo(1), int runHi(9999999);
  int etaMin(1), etaMax(29);
  if (argc>4)  flag   = atoi(argv[4]);
  if (argc>5)  mode   = atoi(argv[5]);
  if (argc>6)  maxDHB = atoi(argv[6]);
  if (argc>7)  maxDHE = atoi(argv[7]);
  if (argc>8)  runLo  = atoi(argv[8]);
  if (argc>9)  runHi  = atoi(argv[9]);
  if (argc>8)  runLo  = atoi(argv[8]);
  if (argc>9)  runHi  = atoi(argv[9]);
  if (argc>10) etaMin = atoi(argv[10]);
  if (argc>11) etaMax = atoi(argv[11]);
  char treeName[400];
  sprintf (treeName, "%s/TREE", processName);
  std::cout << "try to create a chain for " << treeName << std::endl;
  TChain *chain    = new TChain(treeName);

  if (FillChain(chain, inputFileList)) {

    HBHEMuonOfflineAnalyzer* tree = new HBHEMuonOfflineAnalyzer(chain, outFileName, flag, mode, maxDHB, maxDHE, runLo, runHi, etaMin, etaMax);
    tree->Loop();
    
  } 
  return 0;
}

Bool_t FillChain(TChain *chain, const char* inputFileList) {

  ifstream infile(inputFileList);
  if (!infile.is_open()) {
    std::cerr << "** ERROR: Can't open '" << inputFileList << "' for input" << std::endl;
    return kFALSE;
  }

  std::cout << "TreeUtilities : FillChain " << std::endl;
  char buffer[255];
  while (infile) {
    infile.getline(buffer, 255);                // delim defaults to '\n'
    if (!infile.good()) break;
    std::cout << "Adding " << buffer << " to chain" << std::endl;
    chain->Add(buffer);
  }
  std::cout << "No. of Entries in this tree : " << chain->GetEntries() << std::endl;
  infile.close();
  return kTRUE;
}
