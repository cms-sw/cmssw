#ifndef _PlotValidation_
#define _PlotValidation_

#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TDirectory.h"
#include "TString.h"
#include "TEfficiency.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TStyle.h"

#include <string>
#include <vector>
#include <map>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

typedef std::vector<Float_t> FltVec;
typedef std::vector<FltVec> FltVecVec;
typedef std::vector<Double_t> DblVec;
typedef std::vector<DblVec> DblVecVec;
typedef std::vector<Int_t> IntVec;
typedef std::vector<TString> TStrVec;

typedef std::vector<TBranch*> TBrRefVec;
typedef std::vector<TBrRefVec> TBrRefVecVec;
typedef std::vector<TDirectory*> TDirRefVec;

typedef std::map<TString, TH1F*> TH1FRefMap;
typedef std::map<TString, TEfficiency*> TEffRefMap;

struct EffStruct {
  EffStruct() {}
  ~EffStruct() {}

  Float_t passed_;
  Float_t total_;

  Float_t eff_;
  Float_t elow_;
  Float_t eup_;
};

class PlotValidation {
public:
  PlotValidation(const TString& inName,
                 const TString& outName,
                 const Bool_t cmsswComp,
                 const int algo,
                 const Bool_t mvInput,
                 const Bool_t rmSuffix,
                 const Bool_t saveAs,
                 const TString& outType);
  ~PlotValidation();

  // setup functions
  void SetupStyle();
  void SetupBins();
  void SetupVariableBins(const std::string& s_bins, DblVec& bins);
  void SetupFixedBins(const UInt_t nBins, const Double_t low, const Double_t high, DblVec& bins);
  void SetupCommonVars();

  // main call
  void Validation(int algo = 0);
  void PlotEffTree(int algo = 0);
  void PlotFRTree(int algo = 0);
  void PrintTotals(int algo = 0);

  // output functions
  template <typename T>
  void DrawWriteSavePlot(T*& plot, TDirectory*& subdir, const TString& subdirname, const TString& option);

  // helper functions
  void MakeOutDir(const TString& outdirname);
  void GetTotalEfficiency(const TEfficiency* eff, EffStruct& effs);
  TDirectory* MakeSubDirs(const TString& subdirname);
  void MoveInput();

private:
  // input+output config
  const TString fInName;
  const Bool_t fCmsswComp;
  const Bool_t fMvInput;
  const Bool_t fRmSuffix;
  const Bool_t fSaveAs;
  const TString fOutType;

  const int fAlgo;

  // main input
  TFile* fInRoot;
  TTree* efftree;
  TTree* frtree;

  // binning for rate plots
  DblVec fPtBins;
  DblVec fEtaBins;
  DblVec fPhiBins;
  DblVec fNLayersBins;

  // binning for track quality hists
  DblVec fNHitsBins;
  DblVec fFracHitsBins;
  DblVec fScoreBins;

  // binning for diff hists
  DblVec fDNHitsBins;
  DblVec fDInvPtBins;
  DblVec fDPhiBins;
  DblVec fDEtaBins;

  // rate vars
  TStrVec fVars;
  TStrVec fSVars;
  TStrVec fSUnits;
  UInt_t fNVars;

  TString fSVarPt;
  TString fSUnitPt;

  // rate bins
  DblVecVec fVarBins;

  // track collections
  TStrVec fTrks;
  TStrVec fSTrks;
  UInt_t fNTrks;

  // pt cuts
  FltVec fPtCuts;
  TStrVec fSPtCuts;
  TStrVec fHPtCuts;
  UInt_t fNPtCuts;

  // track quality plots
  TStrVec fTrkQual;
  TStrVec fSTrkQual;
  UInt_t fNTrkQual;

  // reference related strings
  TString fSRefTitle;
  TString fSRefVar;
  TString fSRefMask;
  TString fSRefVarTrk;
  TString fSRefDir;
  TString fSRefOut;

  // output variables
  TString fOutName;
  TFile* fOutRoot;
};

#endif
