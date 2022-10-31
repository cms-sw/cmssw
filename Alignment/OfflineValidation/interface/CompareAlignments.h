#ifndef Alignment_OfflineValidation_CompareAlignments_h
#define Alignment_OfflineValidation_CompareAlignments_h

#include <string.h>
#include <cstring>
#include "TH1.h"
#include "TTree.h"
#include "TKey.h"
#include "TMath.h"
#include "Riostream.h"
#include <vector>
#include <sstream>
#include "TCanvas.h"
#include "TLegend.h"
#include "TROOT.h"
#include "TPaveStats.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TStyle.h"
#include "TEnv.h"
#include "TChain.h"
#include "TFile.h"

#include "Alignment/OfflineValidation/interface/TkAlStyle.h"

using namespace std;

class CompareAlignments{
    private:
        TList *FileList;
        TList *LabelList;
        TFile *Target;
        std::vector< std::string > lowestlevels;
        std::vector<int> theColors;
        std::vector<int> theStyles;
        std::vector<int> phases;

        std::string outPath;

        void MergeRootfile( TDirectory *target, TList *sourcelist, TList *labellist, bool bigtext );
        void nicePad(Int_t logx,Int_t logy);
        void SetMinMaxRange(TObjArray *hists);
        void ColourStatsBoxes(TObjArray *hists);

    public:
        CompareAlignments(const std::string& outPath): outPath(outPath){}
        void doComparison(TString namesandlabels, TString legendheader = "", TString lefttitle = "", TString righttitle = "", bool bigtext = false);
};

#endif
