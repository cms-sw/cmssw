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
  if (argc < 4) {
    std::cerr << "Please give 15 arguments \n"
              << "InputFileName"
              << "\n"
              << "outputFileName"
              << "\n"
              << "flag (default 0)\n"
              << "mode (default 3)\n"
              << "cutMu: muon type (default 0)\n"
              << "minimum muon momentum (default 10)\n"
              << "maximum entries to be processed: -1 means all (default -1)\n"
              << "override all constraints in storage (default 0)\n"
              << "maxDHB (default 4)\n"
              << "maxDHE (default 7)\n"
              << "runLo (default 0)\n"
              << "runHi (default 99999999)\n"
              << "etaMin (default 1)\n"
              << "etaMax (default 29)\n"
              << "corrFileName"
              << "\n"
              << std::endl;
    return -1;
  }

  const char *inputFileList = argv[1];
  const char *outFileName = argv[2];
  int flag = (argc > 3) ? atoi(argv[3]) : 0;
  int mode = (argc > 4) ? atoi(argv[4]) : 3;
  int cutMu = (argc > 5) ? atoi(argv[5]) : 0;
  float cutP = (argc > 6) ? atof(argv[6]) : 5;
  int nevMax = (argc > 7) ? atoi(argv[7]) : -1;
  int over = (argc > 8) ? atoi(argv[8]) : 0;
  int maxDHB = (argc > 9) ? atoi(argv[9]) : 4;
  int maxDHE = (argc > 10) ? atoi(argv[10]) : 7;
  int runLo = (argc > 11) ? atoi(argv[11]) : 0;
  int runHi = (argc > 12) ? atoi(argv[12]) : 99999999;
  int etaMin = (argc > 13) ? atoi(argv[13]) : 1;
  int etaMax = (argc > 14) ? atoi(argv[14]) : 29;
  const char *corrFileName = (argc > 15) ? argv[15] : "";
  std::cout << "Input File List " << inputFileList << std::endl
            << "Output FIle Name " << outFileName << std::endl
            << "Correction File Name " << corrFileName << std::endl
            << "Flag " << flag << " Mode " << mode << std::endl
            << "Muon type " << cutMu << " Minimum muon momentum " << cutP << std::endl
            << "Maximum # of entries to process " << nevMax << " Override " << over << std::endl
            << "Max Depth (HB) " << maxDHB << " (HE) " << maxDHE << std::endl
            << "Run (low) " << runLo << " (High) " << runHi << std::endl
            << "Eta (min) " << etaMin << " (max) " << etaMax << std::endl;

  HBHEMuonOfflineAnalyzer *tree = new HBHEMuonOfflineAnalyzer(inputFileList,
                                                              outFileName,
                                                              corrFileName,
                                                              flag,
                                                              mode,
                                                              maxDHB,
                                                              maxDHE,
                                                              cutMu,
                                                              cutP,
                                                              nevMax,
                                                              over,
                                                              runLo,
                                                              runHi,
                                                              etaMin,
                                                              etaMax,
                                                              false);
  tree->Loop();
  delete tree;

  return 0;
}
