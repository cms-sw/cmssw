#ifndef Alignment_OfflineValidation_CompareAlignments_h
#define Alignment_OfflineValidation_CompareAlignments_h

#include "Riostream.h"
#include "TCanvas.h"
#include "TChain.h"
#include "TEnv.h"
#include "TFile.h"
#include "TH1.h"
#include "TKey.h"
#include "TLegend.h"
#include "TMath.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TPaveStats.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TTree.h"
#include <cstring>
#include <sstream>
#include <vector>

#include "Alignment/OfflineValidation/interface/TkAlStyle.h"

class CompareAlignments {
private:
  TList *FileList;
  TList *LabelList;
  TFile *Target;
  std::vector<std::string> lowestlevels;
  std::vector<int> theColors;
  std::vector<int> theStyles;
  std::vector<int> phases;

  std::string outPath;

  void MergeRootfile(TDirectory *target, TList *sourcelist, TList *labellist, bool bigtext);
  void nicePad(Int_t logx, Int_t logy);
  void SetMinMaxRange(TObjArray *hists);
  void ColourStatsBoxes(TObjArray *hists);

public:
  CompareAlignments(const std::string &outPath) : outPath(outPath) {}
  void doComparison(TString namesandlabels,
                    TString legendheader = "",
                    TString lefttitle = "",
                    TString righttitle = "",
                    PublicationStatus status = INTERNAL,
                    bool bigtext = false);
};

#endif
