//////////////////////////////////////////////////////////
// Header with CalibTree class
// for L3 iterative procedure
// implemented in L3_IsoTrackCalibration.C
// 
// version 2.0 January 2016
// M. Chadeeva
//////////////////////////////////////////////////////////

#include <TSystem.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TTree.h>
#include <TH1.h>
#include <TGraph.h>
#include <TGraphErrors.h>
#include <TProfile.h>
#include <TLegend.h>
#include <TString.h>
#include <TF1.h>

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <utility>


//**********************************************************
// Class with TTree containing parameters of selected events
//**********************************************************
class CalibTree {
public :
  TChain          *fChain;   //!pointer to the analyzed TTree
  //TChain          *inChain;   //!pointer to the analyzed TChain
  Int_t           fCurrent; //!current Tree number in a TChain

  // Declaration of leaf types
  Int_t           t_Run;
  Int_t           t_Event;
  Int_t           t_nVtx;
  Int_t           t_nTrk;
  Double_t        t_EventWeight;
  Double_t        t_p;
  Double_t        t_pt;
  Int_t           t_ieta;
  Double_t        t_phi;
  Double_t        t_eMipDR;
  Double_t        t_eHcal;
  Double_t        t_eHcal10;
  Double_t        t_eHcal30;
  Double_t        t_hmaxNearP;
  Bool_t          t_selectTk;
  Bool_t          t_qltyMissFlag;
  Bool_t          t_qltyPVFlag;
/*
  Double_t        t_l1pt;
  Double_t        t_l1eta;
  Double_t        t_l1phi;
  Double_t        t_l3pt;
  Double_t        t_l3eta;
  Double_t        t_l3phi;
*/
  Double_t        t_mindR1;
  Double_t        t_mindR2;
  std::vector<unsigned int> *t_DetIds;
  //std::vector<unsigned int> *t_DetIds1;
  //std::vector<unsigned int> *t_DetIds3;
  std::vector<double>  *t_HitEnergies;
  //std::vector<double>  *t_HitEnergies1;
  //std::vector<double>  *t_HitEnergies3;
  
  // List of branches
  TBranch        *b_t_Run;   //!
  TBranch        *b_t_Event;   //!
  TBranch        *b_t_nVtx;
  TBranch        *b_t_nTrk;
  TBranch        *b_t_EventWeight;   //!
  TBranch        *b_t_p;   //!
  TBranch        *b_t_pt;   //!
  TBranch        *b_t_ieta;   //!
  TBranch        *b_t_phi;   //!
  TBranch        *b_t_eMipDR;   //!
  TBranch        *b_t_eHcal;   //!
  TBranch        *b_t_eHcal10;   //!
  TBranch        *b_t_eHcal30;   //!
  TBranch        *b_t_hmaxNearP;   //!
  TBranch        *b_t_selectTk;   //!
  TBranch        *b_t_qltyMissFlag;   //!
  TBranch        *b_t_qltyPVFlag;   //!
/*
  TBranch        *b_t_l1pt;   //!
  TBranch        *b_t_l1eta;   //!
  TBranch        *b_t_l1phi;   //!
  TBranch        *b_t_l3pt;   //!
  TBranch        *b_t_l3eta;   //!
  TBranch        *b_t_l3phi;   //!
*/
  TBranch        *b_t_mindR1;   //!
  TBranch        *b_t_mindR2;   //!
  TBranch        *b_t_DetIds;   //!
  //TBranch        *b_t_DetIds1;   //!
  //TBranch        *b_t_DetIds3;   //!
  TBranch        *b_t_HitEnergies;   //!
  //TBranch        *b_t_HitEnergies1;   //!
  //TBranch        *b_t_HitEnergies3;   //!

  //--- constructor & destructor
  //CalibTree(TTree *tree=0);
  CalibTree(TChain *tree,
	    double min_enrHcal, double min_pt,
	    double lim_mipEcal, double lim_charIso,
	    double min_trackMom, double max_trackMom);
  virtual ~CalibTree();
  
  //--- functions
  virtual Int_t      GetEntry(Long64_t entry);
  virtual Long64_t   LoadTree(Long64_t entry);
  //virtual void     Init(TTree *tree);
  virtual void       Init(TChain *tree);
  virtual Bool_t     Notify();
  virtual Int_t      firstLoop(unsigned, bool, unsigned);
  virtual Double_t   loopForIteration(unsigned, unsigned, unsigned);
  virtual Double_t   lastLoop(unsigned, unsigned, bool, unsigned);
  Bool_t             goodTrack(int);
  Bool_t             getFactorsFromFile(std::string, unsigned);
  unsigned int       saveFactorsInFile(std::string);
  Bool_t             openOutputRootFile(std::string);

  //--- variables for iterations
  Double_t referenceResponse;
  Double_t referenceResponseHB;
  Double_t referenceResponseTR;
  Double_t referenceResponseHE;
  Double_t maxZtestFromWeights;
  Double_t maxSys2StatRatio;
  int maxNumOfTracksForIeta;
  std::map<unsigned int, double> factors;
  std::map<unsigned int, double> uncFromWeights;
  std::map<unsigned int, double> uncFromDeviation;
  std::map<unsigned int, int> subDetector_trk;
  std::map<unsigned int, int> subDetector_final;
  std::map<unsigned int, int> nTrks;
  std::map<unsigned int, int> nSubdetInEvent;
  std::map<unsigned int, int> nPhiMergedInEvent;
  std::map<unsigned int, double> sumOfResponse;
  std::map<unsigned int, double> sumOfResponseSquared;

  //--- variables for selection
  double minEnrHcal;
  double minTrackPt;
  double minTrackMom;
  double maxTrackMom;
  double limMipEcal;
  double limCharIso;
  double constForFlexSel;
  
  //--- variables for plotting
  TFile *foutRootFile;
  TH1F* e2p_init;
  TH1F* e2pHB_init;
  TH1F* e2pTR_init;
  TH1F* e2pHE_init;
  TH1F* e2p_last;
  TH1F* e2pHB_last;
  TH1F* e2pTR_last;
  TH1F* e2pHE_last;
  
  TH1F* ieta_lefttail;
  TH1F* ieta_righttail;
  /*
  TProfile* deltaVSieta;
  TProfile* deltaNorm;
  TProfile* eHcalDeltaHB;
  TProfile* eHcalDeltaTR;
  TProfile* eHcalDeltaHE;
  TProfile* eHcalDeltaHEfwd;
  TProfile* frespDeltaHB;
  TProfile* frespDeltaTR;
  TProfile* frespDeltaHE;
  TProfile* frespDeltaHEfwd;
  */
};

//**********************************************************
// CalibTree constructor
//**********************************************************
//CalibTree::CalibTree(TTree *tree) : fChain(0) {
CalibTree::CalibTree(TChain *tree,
		     double min_enrHcal,
		     double min_pt,
		     double lim_mipEcal,
		     double lim_charIso,
		     double min_trackMom,
		     double max_trackMom )
{ //: fChain(0) {
  // if parameter tree is not specified (or zero), connect the file
  // used to generate this class and read the Tree.
  if (tree == 0) {
    TFile *f = (TFile*)gROOT->GetListOfFiles()->FindObject("output.root");
    if (!f || !f->IsOpen()) {
      f = new TFile("output.root");
    }
    TDirectory * dir = (TDirectory*)f->Get("IsoTrackCalibration");
    dir->GetObject("CalibTree",tree);
  }
  Init(tree);

  referenceResponse = 1;
  maxNumOfTracksForIeta = 0;
  // initialization of maps
  factors.clear();
  uncFromWeights.clear();
  uncFromDeviation.clear();
  subDetector_trk.clear();
  subDetector_final.clear();
  nTrks.clear();
  nSubdetInEvent.clear();
  nPhiMergedInEvent.clear();
  sumOfResponse.clear();
  sumOfResponseSquared.clear();
  
  // selection
  minEnrHcal = min_enrHcal;
  minTrackPt = min_pt;
  minTrackMom = min_trackMom;
  maxTrackMom = max_trackMom;
  limMipEcal = lim_mipEcal;
  limCharIso = abs(lim_charIso);
  if ( lim_charIso < 0 ) 
    constForFlexSel = log(FLEX_SEL_FIRST_CONST/limCharIso)/FLEX_SEL_SECOND_CONST;
  else constForFlexSel = 0;
}

//**********************************************************
// CalibTree destructor
//**********************************************************
CalibTree::~CalibTree() {

    foutRootFile->cd();
    foutRootFile->Write();
    foutRootFile->Close();

  if (!fChain) return;
  delete fChain->GetCurrentFile();
}

//**********************************************************
// Get entry function
//**********************************************************
Int_t CalibTree::GetEntry(Long64_t entry) {
  // Read contents of entry.
  if (!fChain) return 0;
  return fChain->GetEntry(entry);
}

//**********************************************************
// Load tree function
//**********************************************************
Long64_t CalibTree::LoadTree(Long64_t entry) {
  // Set the environment to read one entry
  if (!fChain) return -5;
  Long64_t centry = fChain->LoadTree(entry);
  if (centry < 0) return centry;
  if (fChain->GetTreeNumber() != fCurrent) {
    fCurrent = fChain->GetTreeNumber();
    Notify();
  }
  return centry;
}

//**********************************************************
// Initialisation of TTree
//**********************************************************
void CalibTree::Init(TChain *tree) {
  // Set object pointer
  t_DetIds = 0;
  //t_DetIds1 = 0;
  //t_DetIds3 = 0;
  t_HitEnergies = 0;
  //t_HitEnergies1 = 0;
  //t_HitEnergies3 = 0;
  // Set branch addresses and branch pointers
  if (!tree) return;
  fChain = tree;
  fCurrent = -1;
  fChain->SetMakeClass(1);
  
  fChain->SetBranchAddress("t_Run", &t_Run, &b_t_Run);
  fChain->SetBranchAddress("t_Event", &t_Event, &b_t_Event);
  fChain->SetBranchAddress("t_nVtx", &t_nVtx, &b_t_nVtx);
  fChain->SetBranchAddress("t_nTrk", &t_nTrk, &b_t_nTrk);
  fChain->SetBranchAddress("t_EventWeight", &t_EventWeight, &b_t_EventWeight);
  fChain->SetBranchAddress("t_p", &t_p, &b_t_p);
  fChain->SetBranchAddress("t_pt", &t_pt, &b_t_pt);
  fChain->SetBranchAddress("t_ieta", &t_ieta, &b_t_ieta);
  fChain->SetBranchAddress("t_phi", &t_phi, &b_t_phi);
  fChain->SetBranchAddress("t_eMipDR", &t_eMipDR, &b_t_eMipDR);
  fChain->SetBranchAddress("t_eHcal", &t_eHcal, &b_t_eHcal);
  fChain->SetBranchAddress("t_eHcal10", &t_eHcal10, &b_t_eHcal10);
  fChain->SetBranchAddress("t_eHcal30", &t_eHcal30, &b_t_eHcal30);
  fChain->SetBranchAddress("t_hmaxNearP", &t_hmaxNearP, &b_t_hmaxNearP);
  fChain->SetBranchAddress("t_selectTk", &t_selectTk, &b_t_selectTk);
  fChain->SetBranchAddress("t_qltyMissFlag", &t_qltyMissFlag, &b_t_qltyMissFlag);
  fChain->SetBranchAddress("t_qltyPVFlag", &t_qltyPVFlag, &b_t_qltyPVFlag);
/*
  fChain->SetBranchAddress("t_l1pt", &t_l1pt, &b_t_l1pt);
  fChain->SetBranchAddress("t_l1eta", &t_l1eta, &b_t_l1eta);
  fChain->SetBranchAddress("t_l1phi", &t_l1phi, &b_t_l1phi);
  fChain->SetBranchAddress("t_l3pt", &t_l3pt, &b_t_l3pt);
  fChain->SetBranchAddress("t_l3eta", &t_l3eta, &b_t_l3eta);
  fChain->SetBranchAddress("t_l3phi", &t_l3phi, &b_t_l3phi);
*/
  fChain->SetBranchAddress("t_mindR1", &t_mindR1, &b_t_mindR1);
  fChain->SetBranchAddress("t_mindR2", &t_mindR2, &b_t_mindR2);
  fChain->SetBranchAddress("t_DetIds", &t_DetIds, &b_t_DetIds);
  //fChain->SetBranchAddress("t_DetIds1", &t_DetIds1, &b_t_DetIds1);
  //fChain->SetBranchAddress("t_DetIds3", &t_DetIds3, &b_t_DetIds3);
  fChain->SetBranchAddress("t_HitEnergies", &t_HitEnergies, &b_t_HitEnergies);
  //fChain->SetBranchAddress("t_HitEnergies1", &t_HitEnergies1, &b_t_HitEnergies1);
  //fChain->SetBranchAddress("t_HitEnergies3", &t_HitEnergies3, &b_t_HitEnergies3);
  Notify();
}

//**********************************************************
// Notification when opening new file
//**********************************************************
Bool_t CalibTree::Notify() {
  // The Notify() function is called when a new file is opened. This
  // can be either for a new TTree in a TChain or when when a new TTree
  // is started when using PROOF. It is normally not necessary to make changes
  // to the generated code, but the routine can be extended by the
  // user if needed. The return value is currently not used.
  
  return kTRUE;
}
//**********************************************************
// Open file and book histograms
//**********************************************************
bool CalibTree::openOutputRootFile(std::string fname)
{
  bool decision = false;
  
  foutRootFile = new TFile(fname.c_str(), "RECREATE");
  if ( foutRootFile != NULL ) decision = true;  
  foutRootFile->cd();

  return decision;
}
//**********************************************************


