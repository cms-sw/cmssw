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

using namespace std;

#include "AnalyzeLepTree.C"

int main(Int_t argc, Char_t *argv[]) {
  if (argc <= 4) {
    std::cerr << "Please give 5 arguments \n"
              << "InputFileName"
              << "\n"
              << "outputRootFileName"
              << "\n"
              << "outputMeanFileName"
              << "\n"
              << "mode (default 0)\n"
              << "modeLHC (default 3)\n"
              << std::endl;
    return -1;
  }

  const char *inputFileName = argv[1];
  const char *outputFileName = argv[2];
  const char *meanFileName = argv[3];
  int mode = (argc > 4) ? atoi(argv[4]) : 0;
  int modeLHC = (argc > 5) ? atoi(argv[5]) : 3;
  int nMax = (argc > 6) ? atoi(argv[6]) : -1;
  bool debug = (argc > 7) ? (atoi(argv[7]) > 0) : false;
  std::cout << "Input File List " << inputFileName << std::endl
            << "Output ROOT File Name " << outputFileName << std::endl
            << "Output Mean File Name " << meanFileName << std::endl
            << "Mode " << mode << " ModeLHC " << modeLHC << std::endl;

  AnalyzeLepTree tree(inputFileName, mode, modeLHC);
  tree.Loop(nMax, debug);
  tree.writeHisto(outputFileName);
  tree.writeMeanError(meanFileName);

  return 0;
}
