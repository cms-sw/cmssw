//  .
// ..: P. Chang, philip@physics.ucsd.edu

#ifndef printutil_cc
#define printutil_cc

// C/C++
#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include <stdarg.h>
#include <functional>
#include <cmath>

// ROOT
#include "TBenchmark.h"
#include "TBits.h"
#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TLeaf.h"
#include "TH1.h"
#include "TH1D.h"
#include "TH2D.h"
#include "TChainElement.h"
#include "TTreeCache.h"
#include "TTreePerfStats.h"
#include "TStopwatch.h"
#include "TSystem.h"
#include "TString.h"
#include "TLorentzVector.h"
#include "Math/LorentzVector.h"
#include "Math/GenVector/PtEtaPhiM4D.h"

#ifdef LorentzVectorPtEtaPhiM4D
typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<float> > LV;
#else
typedef ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<float> > LV;
#endif

namespace RooUtil {

  ///////////////////////////////////////////////////////////////////////////////////////////////
  // Printing functions
  ///////////////////////////////////////////////////////////////////////////////////////////////
  // No namespace given in order to minimize typing
  // (e.g. RooUtil::print v. RooUtil::NAMESPACE::print)
  void clearline(int numchar = 100);
  void print(TString msg = "", const char* fname = "", int flush_before = 0, int flush_after = 0);
  void error(TString msg, const char* fname = "", int is_error = 1);
  void warning(TString msg, const char* fname = "");
  void announce(TString msg = "", int quiet = 0);
  void start(int quiet = 0, int sleep_time = 0);
  void end(int quiet = 0);

  std::string getstr(const LV& lv);
}  // namespace RooUtil

#endif
