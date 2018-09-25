#include <TStyle.h>
#include <TCanvas.h>
#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TProfile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TString.h>
#include <TTree.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "HBHEMuonOfflineAnalyzer.C"

int main(Int_t argc, Char_t *argv[]) {
  
  if (argc<4) {
    std::cerr << "Please give 11 arguments \n"
	      << "InputFileName" << "\n"
	      << "outputFileName" << "\n"
	      << "corrFileName" << "\n"
	      << "flag (default 0)\n"
	      << "mode (default 2)\n"
	      << "maxDHB (default 5)\n"
	      << "maxDHE (default 7)\n"
	      << "runLo (default 0)\n"
	      << "runHi (default 99999999)\n"
	      << "etaMin (default 1)\n"
	      << "etaMax (default 29)\n"
	      << std::endl;
    return -1;
  }

  const char *inputFileList   = argv[1];
  const char *outFileName     = argv[2];
  const char *corrFileName    = argv[3];
  std::cout << "Input File List "  << inputFileList << std::endl
	    << "Output FIle Name " << outFileName << std::endl
	    << "Correction File Name " << corrFileName << std::endl;
  
  int flag(0), mode(2), maxDHB(5), maxDHE(7), runLo(0), runHi(9999999);
  int etaMin(1), etaMax(29);
  if (argc>4)  flag   = atoi(argv[4]);
  if (argc>5)  mode   = atoi(argv[5]);
  if (argc>6)  maxDHB = atoi(argv[6]);
  if (argc>7)  maxDHE = atoi(argv[7]);
  if (argc>8)  runLo  = atoi(argv[8]);
  if (argc>9)  runHi  = atoi(argv[9]);
  if (argc>10) etaMin = atoi(argv[10]);
  if (argc>11) etaMax = atoi(argv[11]);

  HBHEMuonOfflineAnalyzer* tree = 
    new HBHEMuonOfflineAnalyzer(inputFileList, outFileName, corrFileName,
				flag, mode, maxDHB, maxDHE, runLo, runHi,
				etaMin, etaMax, false);
  tree->Loop();

  return 0;
}
