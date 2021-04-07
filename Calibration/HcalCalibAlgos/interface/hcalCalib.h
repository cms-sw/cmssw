////////////////////////////////////////////////////////////////
// This class skeleton has been automatically generated on
// from TTree hcalCalibTree/Tree for IsoTrack Calibration
// with ROOT version 5.17/02
//
//  TSelector-based code for getting the HCAL resp. correction
//  from physics events. Works for DiJet and IsoTrack calibration.
//
//  Anton Anastassov (Northwestern)
//  Email: aa@fnal.gov
//
//
///////////////////////////////////////////////////////////////

#ifndef hcalCalib_h
#define hcalCalib_h

#include <vector>
#include <map>

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TSelector.h>
#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TSelector.h>

#include "TLorentzVector.h"
#include "TClonesArray.h"
#include "TRefArray.h"

// needed to get cell coordinates
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

class hcalCalib : public TSelector {
public:
  TTree *fChain;  //!pointer to the analyzed TTree or TChain

  UInt_t eventNumber;
  UInt_t runNumber;
  Int_t iEtaHit;
  UInt_t iPhiHit;
  TClonesArray *cells;
  Float_t emEnergy;
  Float_t targetE;
  Float_t etVetoJet;

  Float_t xTrkHcal;
  Float_t yTrkHcal;
  Float_t zTrkHcal;
  Float_t xTrkEcal;
  Float_t yTrkEcal;
  Float_t zTrkEcal;

  TLorentzVector *tagJetP4;
  TLorentzVector *probeJetP4;

  Float_t tagJetEmFrac;
  Float_t probeJetEmFrac;

  // List of branches
  TBranch *b_eventNumber;  //!
  TBranch *b_runNumber;    //!
  TBranch *b_iEtaHit;      //!
  TBranch *b_iPhiHit;      //!
  TBranch *b_cells;        //!
  TBranch *b_emEnergy;     //!
  TBranch *b_targetE;      //!
  TBranch *b_etVetoJet;    //!

  TBranch *b_xTrkHcal;
  TBranch *b_yTrkHcal;
  TBranch *b_zTrkHcal;
  TBranch *b_xTrkEcal;
  TBranch *b_yTrkEcal;
  TBranch *b_zTrkEcal;

  TBranch *b_tagJetEmFrac;    //!
  TBranch *b_probeJetEmFrac;  //!

  TBranch *b_tagJetP4;    //!
  TBranch *b_probeJetP4;  //!

  hcalCalib(TTree * /*tree*/ = nullptr) {}
  ~hcalCalib() override {}
  Int_t Version() const override { return 2; }
  void Begin(TTree *tree) override;
  //   virtual void    SlaveBegin(TTree *tree);
  void Init(TTree *tree) override;
  Bool_t Notify() override;
  Bool_t Process(Long64_t entry) override;
  Int_t GetEntry(Long64_t entry, Int_t getall = 0) override {
    return fChain ? fChain->GetTree()->GetEntry(entry, getall) : 0;
  }
  void SetOption(const char *option) override { fOption = option; }
  void SetObject(TObject *obj) override { fObject = obj; }
  void SetInputList(TList *input) override { fInput = input; }
  TList *GetOutputList() const override { return fOutput; }
  //   virtual void    SlaveTerminate();
  void Terminate() override;

  //------------ CUTS ---------------
  Float_t MIN_TARGET_E;
  Float_t MAX_TARGET_E;

  Float_t MIN_CELL_E;
  Float_t MIN_EOVERP;
  Float_t MAX_EOVERP;
  Float_t MAX_TRK_EME;

  Float_t MAX_ET_THIRD_JET;
  Float_t MIN_DPHI_DIJETS;

  Bool_t SUM_DEPTHS;
  Bool_t SUM_SMALL_DEPTHS;
  Bool_t COMBINE_PHI;

  Int_t HB_CLUSTER_SIZE;
  Int_t HE_CLUSTER_SIZE;

  Bool_t USE_CONE_CLUSTERING;
  Float_t MAX_CONE_DIST;

  Int_t CALIB_ABS_IETA_MAX;
  Int_t CALIB_ABS_IETA_MIN;

  Float_t MAX_PROBEJET_EMFRAC;
  Float_t MAX_TAGJET_EMFRAC;
  Float_t MAX_TAGJET_ABSETA;
  Float_t MIN_TAGJET_ET;

  Float_t MIN_PROBEJET_ABSETA;

  TString CALIB_TYPE;    // "ISO_TRACK" or "DI_JET"
  TString CALIB_METHOD;  // L3, matrix inversion, everage then matrix inversion,...

  TString PHI_SYM_COR_FILENAME;
  Bool_t APPLY_PHI_SYM_COR_FLAG;

  TString OUTPUT_COR_COEF_FILENAME;
  TString HISTO_FILENAME;

  const CaloGeometry *theCaloGeometry;
  const HcalTopology *topo_;

  void SetMinTargetE(Float_t e) { MIN_TARGET_E = e; }
  void SetMaxTargetE(Float_t e) { MAX_TARGET_E = e; }
  void SetSumDepthsFlag(Bool_t b) { SUM_DEPTHS = b; }
  void SetSumSmallDepthsFlag(Bool_t b) { SUM_SMALL_DEPTHS = b; }
  void SetCombinePhiFlag(Bool_t b) { COMBINE_PHI = b; }
  void SetMinCellE(Float_t e) { MIN_CELL_E = e; }
  void SetMinEOverP(Float_t e) { MIN_EOVERP = e; }
  void SetMaxEOverP(Float_t e) { MAX_EOVERP = e; }
  void SetMaxTrkEmE(Float_t e) { MAX_TRK_EME = e; }
  void SetCalibType(const TString &s) { CALIB_TYPE = s; }
  void SetCalibMethod(const TString &s) { CALIB_METHOD = s; }
  void SetHbClusterSize(Int_t i) { HB_CLUSTER_SIZE = i; }
  void SetHeClusterSize(Int_t i) { HE_CLUSTER_SIZE = i; }

  void SetUseConeClustering(Bool_t b) { USE_CONE_CLUSTERING = b; }
  void SetConeMaxDist(Float_t d) { MAX_CONE_DIST = d; }

  void SetCalibAbsIEtaMax(Int_t i) { CALIB_ABS_IETA_MAX = i; }
  void SetCalibAbsIEtaMin(Int_t i) { CALIB_ABS_IETA_MIN = i; }
  void SetMaxEtThirdJet(Float_t et) { MAX_ET_THIRD_JET = et; }
  void SetMinDPhiDiJets(Float_t dphi) { MIN_DPHI_DIJETS = dphi; }
  void SetApplyPhiSymCorFlag(Bool_t b) { APPLY_PHI_SYM_COR_FLAG = b; }
  void SetPhiSymCorFileName(const TString &filename) { PHI_SYM_COR_FILENAME = filename; }
  void SetMaxProbeJetEmFrac(Float_t f) { MAX_PROBEJET_EMFRAC = f; }
  void SetMaxTagJetEmFrac(Float_t f) { MAX_TAGJET_EMFRAC = f; }
  void SetMaxTagJetAbsEta(Float_t e) { MAX_TAGJET_ABSETA = e; }
  void SetMinTagJetEt(Float_t e) { MIN_TAGJET_ET = e; }
  void SetMinProbeJetAbsEta(Float_t e) { MIN_PROBEJET_ABSETA = e; }
  void SetOutputCorCoefFileName(const TString &filename) { OUTPUT_COR_COEF_FILENAME = filename; }
  void SetHistoFileName(const TString &filename) { HISTO_FILENAME = filename; }

  void SetCaloGeometry(const CaloGeometry *g, const HcalTopology *topo) {
    theCaloGeometry = g;
    topo_ = topo;
  }

  void GetCoefFromMtrxInvOfAve();

  Bool_t ReadPhiSymCor();

  void makeTextFile();

  // --------- containers passed to minimizers ----------------
  std::vector<std::vector<Float_t> > cellEnergies;
  std::vector<std::vector<UInt_t> > cellIds;
  std::vector<std::pair<Int_t, UInt_t> > refIEtaIPhi;  // centroid of jet or hottest tower iEta, iPhi
  std::vector<Float_t> targetEnergies;

  std::map<UInt_t, Float_t> phiSymCor;  // holds the phi symmetry corrections read from the file

  std::map<UInt_t, Float_t> solution;  // correction coef: solution from L3, holds final coef for hybrid methods as well
  std::map<Int_t, Float_t>
      iEtaCoefMap;  // correction coef: from matrix inversion AFTER averaging, also intermediate results for hybrid methods

  //   ClassDef(hcalCalib,0);
};

#endif

#ifdef hcalCalib_cxx
inline void hcalCalib::Init(TTree *tree) {
  // The Init() function is called when the selector needs to initialize
  // a new tree or chain. Typically here the branch addresses and branch
  // pointers of the tree will be set.
  // It is normaly not necessary to make changes to the generated
  // code, but the routine can be extended by the user if needed.
  // Init() will be called many times when running on PROOF
  // (once per file to be processed).

  // Set object pointer
  cells = nullptr;
  tagJetP4 = nullptr;
  probeJetP4 = nullptr;

  // Set branch addresses and branch pointers
  if (!tree)
    return;
  fChain = tree;

  //      fChain->SetMakeClass(1);

  fChain->SetBranchAddress("eventNumber", &eventNumber, &b_eventNumber);
  fChain->SetBranchAddress("runNumber", &runNumber, &b_runNumber);
  fChain->SetBranchAddress("iEtaHit", &iEtaHit, &b_iEtaHit);
  fChain->SetBranchAddress("iPhiHit", &iPhiHit, &b_iPhiHit);
  fChain->SetBranchAddress("cells", &cells, &b_cells);
  fChain->SetBranchAddress("emEnergy", &emEnergy, &b_emEnergy);
  fChain->SetBranchAddress("targetE", &targetE, &b_targetE);
  fChain->SetBranchAddress("etVetoJet", &etVetoJet, &b_etVetoJet);

  fChain->SetBranchAddress("xTrkHcal", &xTrkHcal, &b_xTrkHcal);
  fChain->SetBranchAddress("yTrkHcal", &yTrkHcal, &b_yTrkHcal);
  fChain->SetBranchAddress("zTrkHcal", &zTrkHcal, &b_zTrkHcal);
  fChain->SetBranchAddress("xTrkEcal", &xTrkEcal, &b_xTrkEcal);
  fChain->SetBranchAddress("yTrkEcal", &yTrkEcal, &b_yTrkEcal);
  fChain->SetBranchAddress("zTrkEcal", &zTrkEcal, &b_zTrkEcal);

  fChain->SetBranchAddress("tagJetEmFrac", &tagJetEmFrac, &b_tagJetEmFrac);
  fChain->SetBranchAddress("probeJetEmFrac", &probeJetEmFrac, &b_probeJetEmFrac);

  fChain->SetBranchAddress("tagJetP4", &tagJetP4, &b_tagJetP4);
  fChain->SetBranchAddress("probeJetP4", &probeJetP4, &b_probeJetP4);
}

inline Bool_t hcalCalib::Notify() {
  // The Notify() function is called when a new file is opened. This
  // can be either for a new TTree in a TChain or when when a new TTree
  // is started when using PROOF. It is normaly not necessary to make changes
  // to the generated code, but the routine can be extended by the
  // user if needed. The return value is currently not used.

  return kTRUE;
}

#endif  // #ifdef hcalCalib_cxx
